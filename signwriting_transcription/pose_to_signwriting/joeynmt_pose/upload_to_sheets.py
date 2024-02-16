import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = "19mluHVMTjIraUWxxJ5FaDFk0nNZOx2fNFsRbo3tc50Q"
SAMPLE_RANGE_NAME = "Sheet1!A2:J"


def upload_line(new_row):
    """Shows basic usage of the Sheets API.
  Prints values from a sample spreadsheet.
  """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("signwriting_transcription/pose_to_signwriting/joeynmt_pose/token.json"):
        creds = Credentials.from_authorized_user_file(
            "signwriting_transcription/pose_to_signwriting/joeynmt_pose/token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "signwriting_transcription/pose_to_signwriting/joeynmt_pose/credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("signwriting_transcription/pose_to_signwriting/joeynmt_pose/token.json", "w",
                  encoding='utf-8') as token:
            token.write(creds.to_json())

    try:
        service = build("sheets", "v4", credentials=creds)
        # Call the Sheets API
        # pylint: disable=no-member
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range=SAMPLE_RANGE_NAME)
            .execute()
        )
        values = result.get("values", [])

        values.append(new_row)

        # Update the sheet with the new values
        body = {"values": values}
        result = (
            sheet.values()
            .update(
                spreadsheetId=SAMPLE_SPREADSHEET_ID,
                range=SAMPLE_RANGE_NAME,
                valueInputOption="USER_ENTERED",
                body=body,
            )
            .execute()
        )

        print("Spreadsheet updated!")
    except HttpError as err:
        print(err)
