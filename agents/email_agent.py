# Email & Calendar Agent with Real Gmail & Google Calendar Integration
# FIXED: Context maintenance, conversation flow, and confirmation system

import os
import json
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Sequence
import base64
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator

# Gmail and Calendar API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/calendar'
]

class GoogleAPIClient:
    """Handles authentication and API calls to Gmail and Calendar"""
    
    def __init__(self):
        self.creds = None
        self.gmail_service = None
        self.calendar_service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Google APIs - FIXED TOKEN EXPIRATION"""
        try:
            # Check for existing token
            if os.path.exists('token.json'):
                try:
                    self.creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                except Exception as e:
                    print(f"Token file corrupted, deleting: {e}")
                    os.remove('token.json')
                    self.creds = None
            
            # If no valid credentials, let user log in
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    try:
                        print("Refreshing expired token...")
                        self.creds.refresh(Request())
                        print("Token refreshed successfully")
                    except Exception as e:
                        print(f"Token refresh failed: {e}")
                        print("Deleting old token and re-authenticating...")
                        if os.path.exists('token.json'):
                            os.remove('token.json')
                        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                        self.creds = flow.run_local_server(port=0)
                else:
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    self.creds = flow.run_local_server(port=0)
                
                # Save credentials for next run
                with open('token.json', 'w') as token:
                    token.write(self.creds.to_json())
            
            # Build services
            self.gmail_service = build('gmail', 'v1', credentials=self.creds)
            self.calendar_service = build('calendar', 'v3', credentials=self.creds)
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            raise
    
    def get_unread_emails(self, max_results=10):
        """Fetch unread emails from Gmail"""
        try:
            results = self.gmail_service.users().messages().list(
                userId='me',
                q='is:unread',
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for msg in messages:
                email_data = self.get_email_details(msg['id'])
                if email_data:
                    emails.append(email_data)
            
            return emails
        except HttpError as error:
            # Auto re-authenticate on 401 error
            if error.resp.status == 401:
                print("Token expired during API call, re-authenticating...")
                self.authenticate()
                return self.get_unread_emails(max_results)
            print(f'Gmail fetch error: {error}')
            return []
    
    def get_email_details(self, msg_id):
        """Get full details of a specific email"""
        try:
            message = self.gmail_service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()
            
            headers = message['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
            
            # Get email body
            body = self.get_email_body(message['payload'])
            
            return {
                'id': msg_id,
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body,
                'thread_id': message['threadId']
            }
        except HttpError as error:
            print(f'Email details error: {error}')
            return None
    
    def get_email_body(self, payload):
        """Extract email body from payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
                elif 'parts' in part:
                    body = self.get_email_body(part)
                    if body:
                        break
        elif 'body' in payload and 'data' in payload['body']:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body
    
    def send_email(self, to, subject, body, thread_id=None):
        """Send an email reply"""
        try:
            message = f"To: {to}\nSubject: Re: {subject}\n\n{body}"
            raw_message = base64.urlsafe_b64encode(message.encode('utf-8')).decode('utf-8')
            
            send_message = {'raw': raw_message}
            if thread_id:
                send_message['threadId'] = thread_id
            
            sent = self.gmail_service.users().messages().send(
                userId='me',
                body=send_message
            ).execute()
            
            return True, sent['id']
        except HttpError as error:
            print(f'Send email error: {error}')
            return False, str(error)
    
    def mark_as_read(self, msg_id):
        """Mark email as read"""
        try:
            self.gmail_service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            return True
        except HttpError as error:
            print(f'Mark as read error: {error}')
            return False
    
    def create_calendar_event(self, summary, start_time, end_time, description="", attendees=None):
        """Create a Google Calendar event"""
        try:
            event = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
            }
            
            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]
            
            event = self.calendar_service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            return True, event.get('htmlLink')
        except HttpError as error:
            print(f'Create calendar event error: {error}')
            return False, str(error)
    
    def get_free_busy(self, start_time, end_time):
        """Check calendar availability"""
        try:
            body = {
                "timeMin": start_time.isoformat() + 'Z',
                "timeMax": end_time.isoformat() + 'Z',
                "items": [{"id": "primary"}]
            }
            
            response = self.calendar_service.freebusy().query(body=body).execute()
            busy_times = response['calendars']['primary']['busy']
            
            return len(busy_times) == 0, busy_times
        except HttpError as error:
            print(f'Free busy check error: {error}')
            return None, str(error)

class EmailAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    email_id: str
    thread_id: str
    email_content: str
    sender: str
    subject: str
    priority: str
    category: str
    action: str
    draft_response: str
    meeting_details: dict
    send_reply: bool
    create_event: bool
    google_client: object

class EmailCalendarAgent:
    """Email and Calendar Management Agent with full Gmail/Calendar integration"""
    
    def __init__(self, llm):
        self.llm = llm
        self.google_client = GoogleAPIClient()
        self.graph = self._build_graph()
        
        # NEW: State management for conversation context
        self.current_email = None  # Stores the selected email
        self.cached_emails = []     # Stores the last fetched email list
        self.draft_ready = None     # Stores the current draft
        self.meeting_ready = None   # Stores meeting details pending confirmation
        self.last_state = None      # Stores the last graph execution state
    
    def _build_graph(self):
        """Create the email processing workflow"""
        workflow = StateGraph(EmailAgentState)
        
        workflow.add_node("triage", self._triage_email)
        workflow.add_node("draft_response", self._draft_response)
        workflow.add_node("extract_meeting", self._extract_meeting_details)
        workflow.add_node("execute_actions", self._execute_actions)
        workflow.add_node("summary", self._create_summary)
        
        workflow.set_entry_point("triage")
        
        workflow.add_conditional_edges(
            "triage",
            self._decide_next_action,
            {
                "extract_meeting": "extract_meeting",
                "draft_response": "draft_response",
                "end": "summary"
            }
        )
        
        workflow.add_edge("extract_meeting", "draft_response")
        workflow.add_edge("draft_response", "execute_actions")
        workflow.add_edge("execute_actions", "summary")
        workflow.add_edge("summary", END)
        
        return workflow.compile()
    
    def _triage_email(self, state: EmailAgentState) -> EmailAgentState:
        """Analyze and categorize the email"""
        prompt = f"""
        Analyze this email and provide:
        1. Priority (High/Medium/Low)
        2. Category (Meeting Request/Question/Information/Action Required/Newsletter/Spam)
        3. Recommended Action (Reply/Schedule/Archive/Forward/Flag)
        
        Email Subject: {state['subject']}
        Email From: {state['sender']}
        Email Content: {state['email_content']}
        
        Respond in JSON format:
        {{
            "priority": "...",
            "category": "...",
            "action": "...",
            "reasoning": "..."
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            state['priority'] = result.get('priority', 'Medium')
            state['category'] = result.get('category', 'Information')
            state['action'] = result.get('action', 'Reply')
            state['messages'].append(AIMessage(content=f"✓ Triage: {result.get('reasoning', '')}"))
        except Exception as e:
            state['priority'] = 'Medium'
            state['category'] = 'Information'
            state['action'] = 'Reply'
            state['messages'].append(AIMessage(content="✓ Triage complete"))
        
        return state
    
    def _extract_meeting_details(self, state: EmailAgentState) -> EmailAgentState:
        """Extract meeting information and check calendar availability"""
        prompt = f"""
        Extract meeting details from this email:
        
        Subject: {state['subject']}
        Content: {state['email_content']}
        
        Provide in JSON format:
        {{
            "has_meeting": true/false,
            "proposed_date": "YYYY-MM-DD",
            "proposed_time": "HH:MM",
            "duration_minutes": 60,
            "topic": "...",
            "needs_response": true/false
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            state['meeting_details'] = result
            
            if result.get('has_meeting'):
                try:
                    date_str = result.get('proposed_date')
                    time_str = result.get('proposed_time')
                    duration = result.get('duration_minutes', 60)
                    
                    start_time = datetime.fromisoformat(f"{date_str}T{time_str}:00")
                    end_time = start_time + timedelta(minutes=duration)
                    
                    is_free, busy_times = self.google_client.get_free_busy(start_time, end_time)
                    
                    result['is_available'] = is_free
                    result['busy_times'] = busy_times
                    state['meeting_details'] = result
                    
                    msg = f"📅 Meeting: {result.get('topic')} - {'Available ✓' if is_free else 'Conflict ✗'}"
                    state['messages'].append(AIMessage(content=msg))
                except Exception as e:
                    state['messages'].append(AIMessage(content=f"📅 Meeting detected but couldn't check availability: {str(e)}"))
            else:
                state['messages'].append(AIMessage(content="No meeting request detected"))
        except Exception as e:
            state['meeting_details'] = {"has_meeting": False}
            state['messages'].append(AIMessage(content="No meeting details found"))
        
        return state
    
    def _draft_response(self, state: EmailAgentState) -> EmailAgentState:
        """Generate a draft email response"""
        context = f"""
        Original Email Subject: {state['subject']}
        From: {state['sender']}
        Content: {state['email_content']}
        
        Category: {state['category']}
        Priority: {state['priority']}
        """
        
        if state.get('meeting_details', {}).get('has_meeting'):
            mtg = state['meeting_details']
            if mtg.get('is_available'):
                context += f"\n\nMeeting Request: ACCEPTED - Calendar is free at proposed time"
            else:
                context += f"\n\nMeeting Request: DECLINED - Calendar conflict. Suggest alternative times."
        
        prompt = f"""
        {context}
        
        Write a professional, concise, and warm email response.
        - Be helpful and clear
        - If meeting is accepted, confirm the details
        - If meeting conflicts, politely suggest alternatives
        - Match the tone of the original email
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state['draft_response'] = response.content
            state['messages'].append(AIMessage(content="✉️ Draft created"))
        except Exception as e:
            state['messages'].append(AIMessage(content="✉️ Failed to create draft"))
        
        return state
    
    def _decide_next_action(self, state: EmailAgentState) -> str:
        """Router function"""
        if state['category'] == 'Meeting Request':
            return "extract_meeting"
        elif state['action'] in ['Reply', 'Action Required']:
            return "draft_response"
        else:
            return "end"
    
    def _execute_actions(self, state: EmailAgentState) -> EmailAgentState:
        """Execute real actions: send email, create calendar event, mark as read"""
        actions_taken = []
        
        # Send reply if flagged
        if state.get('send_reply') and state.get('draft_response'):
            success, result = self.google_client.send_email(
                to=state['sender'],
                subject=state['subject'],
                body=state['draft_response'],
                thread_id=state.get('thread_id')
            )
            if success:
                actions_taken.append(f"✓ Reply sent (ID: {result})")
            else:
                actions_taken.append(f"✗ Failed to send reply: {result}")
        
        # Create calendar event if meeting accepted
        if state.get('create_event') and state.get('meeting_details', {}).get('has_meeting'):
            mtg = state['meeting_details']
            if mtg.get('is_available'):
                try:
                    date_str = mtg.get('proposed_date')
                    time_str = mtg.get('proposed_time')
                    duration = mtg.get('duration_minutes', 60)
                    
                    start_time = datetime.fromisoformat(f"{date_str}T{time_str}:00")
                    end_time = start_time + timedelta(minutes=duration)
                    
                    success, result = self.google_client.create_calendar_event(
                        summary=mtg.get('topic', state['subject']),
                        start_time=start_time,
                        end_time=end_time,
                        description=f"Meeting with {state['sender']}",
                        attendees=[state['sender']]
                    )
                    
                    if success:
                        actions_taken.append(f"✓ Calendar event created: {result}")
                    else:
                        actions_taken.append(f"✗ Failed to create event: {result}")
                except Exception as e:
                    actions_taken.append(f"✗ Error creating event: {str(e)}")
        
        # Mark as read
        if self.google_client.mark_as_read(state['email_id']):
            actions_taken.append("✓ Marked as read")
        
        if actions_taken:
            state['messages'].append(AIMessage(content="\n".join(actions_taken)))
        
        return state
    
    def _create_summary(self, state: EmailAgentState) -> EmailAgentState:
        """Create final summary"""
        summary = f"""
📧 **Email Processing Complete**

**From:** {state['sender']}
**Subject:** {state['subject']}
**Priority:** {state['priority']}
**Category:** {state['category']}

"""
        
        if state.get('draft_response'):
            summary += f"\n**Draft Response:**\n{state['draft_response']}\n"
        
        if state.get('meeting_details', {}).get('has_meeting'):
            mtg = state['meeting_details']
            summary += f"\n**Meeting Details:**\n"
            summary += f"- Topic: {mtg.get('topic', 'N/A')}\n"
            summary += f"- Date: {mtg.get('proposed_date', 'N/A')}\n"
            summary += f"- Time: {mtg.get('proposed_time', 'N/A')}\n"
            summary += f"- Available: {'Yes ✓' if mtg.get('is_available') else 'No ✗'}\n"
        
        state['messages'].append(AIMessage(content=summary))
        return state
    
    def process(self, user_input: str) -> str:
        """Process email/calendar requests with context awareness"""
        user_lower = user_input.lower().strip()
        
        # CONFIRMATION HANDLERS
        if user_lower in ["yes", "y", "confirm", "send", "send it"]:
            if self.draft_ready:
                success, result = self.google_client.send_email(
                    to=self.current_email['sender'],
                    subject=self.current_email['subject'],
                    body=self.draft_ready,
                    thread_id=self.current_email.get('thread_id')
                )
                msg = f"✅ Reply sent successfully! (ID: {result})" if success else f"❌ Failed to send: {result}"
                if success:
                    self.google_client.mark_as_read(self.current_email['id'])
                    self.current_email = None  # Clear after sending
                self.draft_ready = None
                return msg
            
            elif self.meeting_ready:
                mtg = self.meeting_ready
                try:
                    start_time = datetime.fromisoformat(f"{mtg['proposed_date']}T{mtg['proposed_time']}:00")
                    end_time = start_time + timedelta(minutes=mtg.get('duration_minutes', 60))
                    
                    success, result = self.google_client.create_calendar_event(
                        summary=mtg.get('topic', self.current_email['subject']),
                        start_time=start_time,
                        end_time=end_time,
                        description=f"Meeting with {self.current_email['sender']}",
                        attendees=[self.current_email['sender']]
                    )
                    
                    msg = f"✅ Meeting scheduled! {result}" if success else f"❌ Failed: {result}"
                    self.meeting_ready = None
                    return msg
                except Exception as e:
                    return f"❌ Error scheduling meeting: {str(e)}"
            
            else:
                return "❌ Nothing to confirm. Please select an email first or request a draft."
        
        if user_lower in ["no", "n", "cancel", "skip"]:
            self.draft_ready = None
            self.meeting_ready = None
            return "❌ Cancelled. What would you like to do next?"
        
        # LIST EMAILS
        if any(word in user_lower for word in ["check", "show", "list"]) and any(word in user_lower for word in ["email", "inbox", "unread"]):
            self.cached_emails = self.google_client.get_unread_emails(max_results=20)
            if not self.cached_emails:
                return "📭 No unread emails found."
            
            email_list = ["📬 **Your Unread Emails:**\n"]
            for i, email in enumerate(self.cached_emails, 1):
                email_list.append(f"{i}. **From:** {email['sender']}\n   **Subject:** {email['subject']}\n")
            
            email_list.append("\n💡 Type a number to select an email (e.g., '1' or 'select 3')")
            return "\n".join(email_list)
        
        # SELECT EMAIL BY NUMBER
        match = re.search(r'\b(\d+)\b', user_input)
        if match and not any(word in user_lower for word in ["reply", "draft", "schedule"]):
            try:
                email_index = int(match.group(1)) - 1
                
                # Fetch emails if not cached
                if not self.cached_emails:
                    self.cached_emails = self.google_client.get_unread_emails(max_results=20)
                
                if email_index < 0 or email_index >= len(self.cached_emails):
                    return f"❌ Invalid selection. Please choose 1-{len(self.cached_emails)}"
                
                self.current_email = self.cached_emails[email_index]
                
                # Show full email content
                preview = self.current_email['body'][:500] + ("..." if len(self.current_email['body']) > 500 else "")
                
                return (f"📧 **Email Selected:**\n\n"
                       f"**From:** {self.current_email['sender']}\n"
                       f"**Subject:** {self.current_email['subject']}\n"
                       f"**Date:** {self.current_email['date']}\n\n"
                       f"**Content Preview:**\n{preview}\n\n"
                       f"💡 What would you like to do?\n"
                       f"• Type 'draft reply' to create a response\n"
                       f"• Type 'full content' to see the complete email\n"
                       f"• Type 'analyze' to get AI insights")
            
            except Exception as e:
                return f"❌ Error selecting email: {str(e)}"
        
        # SHOW FULL CONTENT OF CURRENT EMAIL
        if user_lower in ["full", "full content", "show all", "read all"]:
            if not self.current_email:
                return "❌ No email selected. Please select an email first."
            
            return (f"📧 **Full Email:**\n\n"
                   f"**From:** {self.current_email['sender']}\n"
                   f"**Subject:** {self.current_email['subject']}\n"
                   f"**Date:** {self.current_email['date']}\n\n"
                   f"**Full Content:**\n{self.current_email['body']}")
        
        # ANALYZE CURRENT EMAIL
        if "analyze" in user_lower:
            if not self.current_email:
                return "❌ No email selected. Please select an email first."
            
            initial_state = {
                "messages": [],
                "email_id": self.current_email['id'],
                "thread_id": self.current_email['thread_id'],
                "email_content": self.current_email['body'],
                "sender": self.current_email['sender'],
                "subject": self.current_email['subject'],
                "priority": "",
                "category": "",
                "action": "",
                "draft_response": "",
                "meeting_details": {},
                "send_reply": False,
                "create_event": False,
                "google_client": self.google_client
            }
            
            result = self.graph.invoke(initial_state)
            self.last_state = result
            
            analysis = (f"🔍 **Email Analysis:**\n\n"
                       f"**Priority:** {result['priority']}\n"
                       f"**Category:** {result['category']}\n"
                       f"**Recommended Action:** {result['action']}\n")
            
            if result.get('meeting_details', {}).get('has_meeting'):
                mtg = result['meeting_details']
                analysis += (f"\n📅 **Meeting Detected:**\n"
                           f"• Topic: {mtg.get('topic', 'N/A')}\n"
                           f"• Date: {mtg.get('proposed_date', 'N/A')}\n"
                           f"• Time: {mtg.get('proposed_time', 'N/A')}\n"
                           f"• Available: {'✅ Yes' if mtg.get('is_available') else '❌ Conflict'}\n")
            
            return analysis
        
        # DRAFT REPLY FOR CURRENT EMAIL
        if any(word in user_lower for word in ["draft", "reply", "respond"]):
            if not self.current_email:
                return "❌ No email selected. Please select an email first."
            
            initial_state = {
                "messages": [],
                "email_id": self.current_email['id'],
                "thread_id": self.current_email['thread_id'],
                "email_content": self.current_email['body'],
                "sender": self.current_email['sender'],
                "subject": self.current_email['subject'],
                "priority": "",
                "category": "",
                "action": "",
                "draft_response": "",
                "meeting_details": {},
                "send_reply": False,
                "create_event": False,
                "google_client": self.google_client
            }
            
            result = self.graph.invoke(initial_state)
            self.last_state = result
            self.draft_ready = result.get('draft_response', '')
            self.meeting_ready = result.get('meeting_details') if result.get('meeting_details', {}).get('has_meeting') else None
            
            response = f"✉️ **Draft Reply:**\n\n{self.draft_ready}\n\n"
            
            if self.meeting_ready and self.meeting_ready.get('is_available'):
                response += (f"📅 **Meeting to Schedule:**\n"
                           f"• {self.meeting_ready.get('topic')}\n"
                           f"• {self.meeting_ready.get('proposed_date')} at {self.meeting_ready.get('proposed_time')}\n\n")
            
            response += "💡 Type 'yes' to send, 'no' to cancel, or 'edit' to modify"
            
            return response
        
        # DEFAULT HELP
        if not self.cached_emails:
            return ("📧📅 **Email & Calendar Assistant**\n\n"
                   "I can help with:\n"
                   "• Managing your Gmail inbox\n"
                   "• Scheduling meetings and events\n"
                   "• Drafting and sending replies\n\n"
                   "💡 Try: 'Check my emails' to get started")
        else:
            return ("💡 **Available commands:**\n"
                   "• Select an email by number (1, 2, 3...)\n"
                   "• 'draft reply' - Create a response\n"
                   "• 'full content' - See complete email\n"
                   "• 'analyze' - Get AI insights\n"
                   "• 'check emails' - Refresh inbox")