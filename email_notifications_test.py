import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email credentials
sender_email = "td32@cttd.biz"
receiver_email = "8qobjttgs6@pomail.net"
password = "your_password"

# Create a multipart message
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = "Python email test"

# Mail body
body = "This is a test email sent from a Python script."
message.attach(MIMEText(body, "plain"))

# Server setup
server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls() # Secure the connection
server.login(sender_email, password)

# Send the email
server.sendmail(sender_email, receiver_email, message.as_string())

# Quit the server
server.quit()

