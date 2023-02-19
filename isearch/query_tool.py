from datetime import datetime, timezone
import sqlite3
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import numpy as np


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
    embeds: bytes = None

    def parse_text(self):
        if self.text is not None:
            return self.text

        attributed_body = self.attributed_body.decode('utf-8', errors='replace')

        body = ""

        if "NSNumber" in str(attributed_body):
            attributed_body = str(attributed_body).split("NSNumber")[0]
            if "NSString" in attributed_body:
                attributed_body = str(attributed_body).split("NSString")[1]
                if "NSDictionary" in attributed_body:
                    attributed_body = str(attributed_body).split("NSDictionary")[0]
                    attributed_body = attributed_body[6:-12]
                    body = attributed_body

        if not body:
            print("oh no")

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





model = SentenceTransformer('all-MiniLM-L6-v2')

# Fetch all embeddings of the given user's chat
chat_id = 233

def findMessage(msg: str):
    def fetch_message_context(message: Message, context_length: int = 5) -> list[Message]:
        msgs = cur.execute("""

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
Hi! Here's a matching message.

{ctx}
{msg.render()}
    """.strip()


    conn = sqlite3.connect('/Users/kevin/chat_isearch.db')
    conn.execute("""
    attach database '/Users/kevin/chat.db' as imsg;
    """)
    cur = conn.cursor()


    messages = cur.execute("""
    select
        message.date AS message_date,
        message.guid as guid,
        message.text,
        message.attributedBody,
        message.is_from_me,
        handle.id,
        chat.display_name,
        message_embeddings.embed
    from message
        join message_embeddings on message_embeddings.guid = message.guid
        join chat_message_join on chat_message_join.message_id = message.ROWID
        join chat on chat.ROWID = chat_message_join.chat_id
        join handle on handle.ROWID = message.handle_id
    WHERE
        chat_message_join.chat_id == 233
        and (message.text is not null or message.attributedBody is not null)
    order BY
        message.date DESC
    """).fetchall()

    m_objs = [Message(*m) for m in messages]
    embeds = [torch.from_numpy(np.frombuffer(row[7], dtype=np.float32)) for row in messages]
    embeds = torch.stack(embeds)
    print(embeds.shape)

    # Compute cosine-similarits
    query = msg
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_embedding, embeds)

    # Sort the results in decreasing order
    top_indexes = torch.topk(cos_scores, k=5).indices

    returned_messages = []
    for idx in top_indexes:
        returned_messages.append(render_context_window(m_objs[idx], fetch_message_context(m_objs[idx])))

    return returned_messages
