import sendgrid

from sendgrid.helpers.mail import Mail, Email, To, Content
from strategy import stragey_supertrend, get_prices

ticker = 'AAPL'
# Get latest prices
df = get_prices(ticker)
# Create position info from our strategy
super_df = stragey_supertrend(df, interval=120, atr_period=14, multiplier=2)

# Sendgrid options
sg = sendgrid.SendGridAPIClient(api_key="YOUR_SENDGRID_APIKEY")
from_email = Email("from@email.com")
to_email = [To("to@email.com")]
subject = f"Position for {ticker}"
contents = f"Close price for {ticker}: {super_df['close'][-1]}, position: {super_df['position'][-1]}"
content = Content("text/plain", content=contents)
mail = Mail(from_email, to_email, subject, content)

# Get a JSON-ready representation of the Mail object
mail_json = mail.get()

# Send an HTTP POST request to /mail/send
response = sg.client.mail.send.post(request_body=mail_json)