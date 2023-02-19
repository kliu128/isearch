import os
from dotenv import load_dotenv
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from query_tool import findMessage

load_dotenv()
app = Flask(__name__)


@app.route('/bot', methods=['GET', 'POST'])
def bot():
    body = request.values.get('Body', None)
    user = request.values.get('From', '')

    responseArray = findMessage(user)

    resp = MessagingResponse()
    built_string = ""

    for msg in responseArray:
        built_string += (msg + "\n")

    resp.message(built_string)
    return str(resp)


def start_ngrok():
    from twilio.rest import Client
    from pyngrok import ngrok

    url = ngrok.connect(5002).public_url
    print(' * Tunnel URL:', url)
    client = Client()
    client.incoming_phone_numbers.list(
        phone_number=os.environ.get('TWILIO_PHONE_NUMBER'))[0].update(
            sms_url=url + '/bot')


if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        start_ngrok()
    app.run(debug=True, port=5002)
