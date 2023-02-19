from datetime import datetime, timezone
import sqlite3
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

db = sqlite3.connect('/Users/kevin/chat_isearch.db')
db.execute("""
attach database '/Users/kevin/chat.db' as imsg;
""")
cur = db.cursor()
# Enable print logging for queries
# db.set_trace_callback(print)

@dataclass
class Message:
    # Nanoseconds since 2001-01-01 00:00:00 UTC
    date: int
    guid: str
    text: str
    attributed_body: bytes
    is_from_me: bool
    handle_id: str
    display_name: str

    def parse_text(self):
        if self.text is not None:
            return self.text

        attributed_body = self.attributed_body.decode('utf-8', errors='replace')

        if "NSNumber" in str(attributed_body):
            attributed_body = str(attributed_body).split("NSNumber")[0]
            if "NSString" in attributed_body:
                attributed_body = str(attributed_body).split("NSString")[1]
                if "NSDictionary" in attributed_body:
                    attributed_body = str(attributed_body).split("NSDictionary")[0]
                    attributed_body = attributed_body[6:-12]
                    body = attributed_body

        return body

    def parse_date(self) -> datetime:
        date_epoch = datetime.strptime('2001-01-01', '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()
        new_date = self.date / 1e9 + date_epoch
        date = datetime.fromtimestamp(new_date, tz=timezone.utc)
        return date

    def render(self) -> datetime:
        sender = "Me" if self.is_from_me else "Other"
        time_string = self.parse_date().isoformat()
        return f"{time_string} - {sender}: {self.parse_text()}"


def fetch_message_context(message: Message, context_length: int = 5) -> list[Message]:
    msgs = db.execute("""
select
    message.date AS message_date,
	message.guid as guid,
    message.text,
	message.attributedBody,
	message.is_from_me,
	handle.id,
	chat.display_name
FROM
	imsg.chat_message_join
	JOIN message ON message.ROWID == chat_message_join.message_id
    JOIN handle on message.handle_id = handle.ROWID
    JOIN chat on chat.ROWID = chat_message_join.chat_id
WHERE
	imsg.chat_message_join.chat_id IN (
		select
			chat_id
		from
			imsg.chat_message_join
		WHERE
			imsg.chat_message_join.message_id IN (
				select ROWID from imsg.message where imsg.message.guid == ?
			)
	)
    AND message.date < ?
    and (message.text is not null or message.attributedBody is not null)
ORDER BY
	message_date DESC
LIMIT ?
""", (message.guid, message.date, context_length)).fetchall()
    return [Message(*m) for m in msgs]


def render_context_window(msg: Message, context: list[Message]) -> str:
    ctx = "\n".join([m.render() for m in context[::-1]])
    return f"""
A text conversation on iMessage.

Context:
{ctx}

This message:
{msg.render()}
""".strip()


model = SentenceTransformer("all-MiniLM-L6-v2").to("mps")
model._target_device = torch.device("mps")
print("loaded sentence transformer!")

BATCH_SIZE = 512

count = db.execute("""
select
    count(*)
from
	imsg.chat
	-- Find chat IDs
    JOIN chat_message_join ON imsg.chat. "ROWID" = chat_message_join.chat_id
    JOIN message ON chat_message_join.message_id = message. "ROWID"
    JOIN handle on message.handle_id = handle.ROWID
    -- Filter by messages where we haven't calculates message embeddings.
	full outer join message_embeddings on message_embeddings.guid = message.guid
where
	message_embeddings.embed is null
    and (message.text is not null or message.attributedBody is not null)
order by
	message_date desc
""").fetchone()

# print("Found {} messages to embed.".format(count[0]))

with tqdm(total=count[0]) as pbar:
    while True:
        messages = db.execute("""
        select
            message.date AS message_date,
            message.guid as guid,
            message.text,
            message.attributedBody,
            message.is_from_me,
            handle.id,
            chat.display_name
        from
            imsg.chat
            -- Find chat IDs
            JOIN chat_message_join ON imsg.chat. "ROWID" = chat_message_join.chat_id
            JOIN message ON chat_message_join.message_id = message. "ROWID"
            JOIN handle on message.handle_id = handle.ROWID
            -- Filter by messages where we haven't calculates message embeddings.
            full outer join message_embeddings on message_embeddings.guid = message.guid
        where
            message_embeddings.embed is null
            and (message.text is not null or message.attributedBody is not null)
        order by
            message_date desc
        limit ?
        """, (BATCH_SIZE,)).fetchall()

        if len(messages) == 0:
            break

        print("Found {} messages to embed.".format(len(messages)))
        messages = [Message(*m) for m in messages]
        contexts = [fetch_message_context(m) for m in messages]

        embeds = model.encode([render_context_window(m, c) for m, c in zip(messages, contexts)], show_progress_bar=False, convert_to_tensor=True)

        for msg, embed in zip(messages, embeds):
            cur.execute("""
            insert into message_embeddings (guid, embed, model_ver) values (?, ?, 1)
            """, (msg.guid, embed.cpu().numpy().tobytes()))
            db.commit()

        pbar.update(BATCH_SIZE)
