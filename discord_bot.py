from discord import SyncWebhook
from enum import Enum
import os
from dotenv import load_dotenv
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
LEO_USER_ID = os.getenv("LEO_USER_ID")
DAVID_USER_ID = os.getenv("DAVID_USER_ID")

class User(Enum):
    Leo = LEO_USER_ID
    David = DAVID_USER_ID

    def get_discord_id(self):
        return f"<@{self.value}>"


def main():
    send_message("Hello",User.Leo)

def send_message(message,user:User):
    webhook = SyncWebhook.from_url(DISCORD_WEBHOOK_URL)
    webhook.send(f"{user.get_discord_id()} {message}")



if __name__ == '__main__':
    main()