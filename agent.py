from typing import Literal, List, Any
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from prompt_library.prompt import system_prompt
from utils.llms import LLMModel
from toolkit.toolkits import *
from pydantic import BaseModel, Field

class Router(BaseModel):
    """Router to determine next action"""
    next: Literal["information_node", "booking_node", "FINISH"] = Field(
        description="The next node to route to based on the user's query"
    )
    reasoning: str = Field(
        description="Brief explanation for why this routing decision was made"
    )

class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    id_number: int
    next: str
    query: str
    current_reasoning: str

class DoctorAppointmentAgent:
    def __init__(self):
        llm_model = LLMModel()
        self.llm_model = llm_model.get_model()
        print("LLM model Name:-", self.llm_model)
    
    def supervisor_node(self, state: AgentState) -> Command[Literal['information_node', 'booking_node', '__end__']]:
        print("**************************below is my state****************************")
        print(state)
        
        # Count how many times each node has been called
        info_calls = len([m for m in state['messages'] if hasattr(m, 'name') and m.name == 'information_node'])
        booking_calls = len([m for m in state['messages'] if hasattr(m, 'name') and m.name == 'booking_node'])
        
        # Prevent infinite loops
        if info_calls + booking_calls >= 6:
            return Command(goto=END, update={'next': 'FINISH', 'current_reasoning': 'Maximum iterations reached'})
        
        system_message = f"""{system_prompt}

    IMPORTANT ROUTING RULES:
    1. If user asks to "check availability", route to information_node
    2. If user asks to "check AND book", FIRST route to information_node to check availability
    3. After information_node confirms availability, route to booking_node to complete the booking
    4. If booking was completed (success or failure), route to FINISH
    5. If no progress after 3 attempts, route to FINISH

    Current state:
    - User ID: {state['id_number']}
    - Information checks done: {info_calls}
    - Booking attempts done: {booking_calls}
    - Last message was from: {state['messages'][-1].name if hasattr(state['messages'][-1], 'name') else 'user'}

    Respond with JSON: {{"next": "information_node"|"booking_node"|"FINISH", "reasoning": "explanation"}}
    """
        
        messages = [
            {"role": "system", "content": system_message},
        ] + [
            {"role": "assistant" if isinstance(msg, AIMessage) else "user", "content": msg.content}
            for msg in state["messages"]
        ]
        
        print("***********************supervisor messages*****************************************")
        
        try:
            response = self.llm_model.with_structured_output(Router).invoke(messages)
            goto = response.next
            reasoning = response.reasoning
            
        except Exception as e:
            print(f"Structured output error: {e}")
            
            # IMPROVED FALLBACK LOGIC
            last_msg = state['messages'][-1]
            last_content = last_msg.content.lower() if hasattr(last_msg, 'content') else ""
            
            # Check if this is the first user message asking to check and book
            first_msg = state['messages'][0].content.lower() if state['messages'] else ""
            is_check_and_book = 'check' in first_msg and 'book' in first_msg
            
            # If last message is from information_node
            if hasattr(last_msg, 'name') and last_msg.name == 'information_node':
                # Check if availability was confirmed
                if ('available' in last_content or 'slot' in last_content) and \
                ('no available' not in last_content and 'not available' not in last_content):
                    # Availability confirmed, route to booking
                    goto = "booking_node"
                    reasoning = "Fallback: availability confirmed, routing to booking"
                elif 'no available' in last_content or 'not available' in last_content:
                    goto = "FINISH"
                    reasoning = "Fallback: no availability found"
                else:
                    # If user originally wanted to book, go to booking
                    if is_check_and_book and booking_calls == 0:
                        goto = "booking_node"
                        reasoning = "Fallback: proceeding to booking as requested"
                    else:
                        goto = "FINISH"
                        reasoning = "Fallback: information provided"
            
            # If last message is from booking_node, finish
            elif hasattr(last_msg, 'name') and last_msg.name == 'booking_node':
                goto = "FINISH"
                reasoning = "Fallback: booking complete"
            
            # If this is the first message and no nodes called yet
            elif info_calls == 0 and booking_calls == 0:
                if 'check' in first_msg or 'available' in first_msg:
                    goto = "information_node"
                    reasoning = "Fallback: checking availability first"
                elif 'book' in first_msg or 'appointment' in first_msg:
                    goto = "booking_node"
                    reasoning = "Fallback: direct booking request"
                else:
                    goto = "FINISH"
                    reasoning = "Fallback: unclear request"
            
            else:
                goto = "FINISH"
                reasoning = "Fallback: ending conversation"
        
        print(f"********************************goto: {goto}*************************")
        print(f"********************************reasoning: {reasoning}*************************")
            
        if goto == "FINISH":
            goto = END
        
        return Command(
            goto=goto, 
            update={
                'next': goto if goto != END else 'FINISH', 
                'current_reasoning': reasoning
            }
        )

    def information_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        print("*****************called information node************")

        info_system_prompt = """You are a specialized agent to provide information about doctor availability.

    YOUR ONLY TOOLS:
    - check_availability_by_specialization: Check availability by specialization
    - check_availability_by_doctor: Check availability by specific doctor

    CRITICAL FORMATTING RULES:
    1. Date format: DD-MM-YYYY (e.g., 08-08-2024) - TWO digits for day and month
    2. Doctor names MUST be ALL LOWERCASE with spaces (e.g., "emily johnson" NOT "Emily Johnson")
    3. Valid doctor names (use EXACTLY as written):
    kevin anderson, robert martinez, susan davis, daniel miller, sarah wilson,
    michael green, lisa brown, jane smith, emily johnson, john doe
    4. Valid specializations:
    general_dentist, cosmetic_dentist, prosthodontist, pediatric_dentist,
    emergency_dentist, oral_surgeon, orthodontist

    WORKFLOW:
    - If user asks about a specialization (e.g., "general dentist"), use check_availability_by_specialization
    - This will return available doctors with their time slots
    - DO NOT try to book appointments - you only check availability
    - Current year is 2024

    Example:
    User: "check if general dentist available on 8 august 2024"
    Action: check_availability_by_specialization(desired_date="08-08-2024", specialization="general_dentist")
    """
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", info_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        
        information_agent = create_react_agent(
            model=self.llm_model,
            tools=[check_availability_by_doctor, check_availability_by_specialization],
            prompt=prompt_template
        )
        
        try:
            result = information_agent.invoke(state)
            final_message = result["messages"][-1].content
        except Exception as e:
            error_msg = str(e)
            # Make error messages more user-friendly
            if "tool call validation failed" in error_msg and "doctor_name" in error_msg:
                final_message = "I apologize, but I need to check availability by specialization first. Let me search for available general dentists on that date."
                # Retry with specialization check
                try:
                    user_query = state['messages'][0].content if state['messages'] else ""
                    # Extract date from query
                    if "8 august 2024" in user_query.lower() or "08-08-2024" in user_query:
                        result = check_availability_by_specialization.invoke({
                            "desired_date": "08-08-2024",
                            "specialization": "general_dentist"
                        })
                        final_message = result
                except Exception as retry_error:
                    final_message = f"I encountered an error while checking availability: {str(retry_error)}"
            else:
                final_message = f"I encountered an error while checking availability: {error_msg}"
        
        return Command(
            update={
                "messages": [AIMessage(content=final_message, name="information_node")]
            },
            goto="supervisor",
        )

    def booking_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        print("*****************called booking node************")

        # Extract info from previous information_node message
        info_messages = [m for m in state['messages'] if hasattr(m, 'name') and m.name == 'information_node']
        availability_info = info_messages[-1].content if info_messages else ""
        
        # Extract doctor name from availability info
        import re
        doctor_match = re.search(r'Dr\.\s+([A-Za-z\s]+)', availability_info)
        extracted_doctor = doctor_match.group(1).strip().lower() if doctor_match else ""
        
        # Extract time from availability info (looking for patterns like "8:00 PM" or "20:00")
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', availability_info, re.IGNORECASE)
        extracted_time = ""
        if time_match:
            hour = int(time_match.group(1))
            minute = time_match.group(2)
            period = time_match.group(3).upper()
            # Convert to 24-hour format
            if period == 'PM' and hour != 12:
                hour += 12
            elif period == 'AM' and hour == 12:
                hour = 0
            extracted_time = f"{hour:02d}:{minute}"
        
        # Extract date
        date_match = re.search(r'(\d{2}-\d{2}-\d{4})', availability_info)
        extracted_date = date_match.group(1) if date_match else ""
        
        booking_system_prompt = f"""You are a specialized booking agent. Your ONLY job is to book appointments.

    AVAILABLE TOOLS - YOU MUST USE ONE OF THESE:
    1. set_appointment(desired_date, id_number, doctor_name) - Book a new appointment
    2. cancel_appointment(date, id_number, doctor_name) - Cancel an appointment  
    3. reschedule_appointment(old_date, new_date, id_number, doctor_name) - Reschedule an appointment

    CRITICAL: DO NOT try to call any other tools. Only use the 3 tools listed above.

    CURRENT BOOKING CONTEXT:
    - Patient ID: {state['id_number']}
    - Availability check result: {availability_info}
    - Extracted doctor name: {extracted_doctor}
    - Extracted date: {extracted_date}
    - Extracted time: {extracted_time}

    INSTRUCTIONS FOR BOOKING:
    1. The user wants to BOOK an appointment
    2. Use the extracted information above to call set_appointment
    3. Parameters:
    - desired_date: "{extracted_date} {extracted_time}" (format: DD-MM-YYYY HH:MM)
    - id_number: {state['id_number']}
    - doctor_name: "{extracted_doctor}" (must be lowercase)

    VALID DOCTOR NAMES (use exactly as listed):
    kevin anderson, robert martinez, susan davis, daniel miller, sarah wilson,
    michael green, lisa brown, jane smith, emily johnson, john doe

    EXAMPLE:
    If extracted data shows: doctor="Emily Johnson", date="08-08-2024", time="20:00"
    Call: set_appointment(desired_date="08-08-2024 20:00", id_number={state['id_number']}, doctor_name="emily johnson")

    NOW PROCEED TO BOOK THE APPOINTMENT using set_appointment tool with the extracted information.
    """
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", booking_system_prompt),
                ("user", f"Please book the appointment with the information provided. Use set_appointment tool now."),
            ]
        )
        
        booking_agent = create_react_agent(
            model=self.llm_model,
            tools=[set_appointment, cancel_appointment, reschedule_appointment],
            prompt=prompt_template
        )

        try:
            # Create a modified state for the booking agent
            booking_state = {
                "messages": [
                    HumanMessage(content=f"Book appointment for patient {state['id_number']} with Dr. {extracted_doctor} on {extracted_date} at {extracted_time}")
                ]
            }
            
            result = booking_agent.invoke(booking_state)
            final_message = result["messages"][-1].content
            
            # If the result is empty or still has errors, try direct tool invocation
            if not final_message or "error" in final_message.lower():
                if extracted_doctor and extracted_date and extracted_time:
                    try:
                        datetime_str = f"{extracted_date} {extracted_time}"
                        direct_result = set_appointment.invoke({
                            "desired_date": datetime_str,
                            "id_number": state['id_number'],
                            "doctor_name": extracted_doctor
                        })
                        final_message = direct_result
                    except Exception as e:
                        final_message = f"Failed to book appointment: {str(e)}"
                else:
                    final_message = f"Could not extract booking details. Doctor: {extracted_doctor}, Date: {extracted_date}, Time: {extracted_time}"
                    
        except Exception as e:
            error_msg = str(e)
            
            # Fallback: Try direct tool invocation
            if extracted_doctor and extracted_date and extracted_time:
                try:
                    datetime_str = f"{extracted_date} {extracted_time}"
                    direct_result = set_appointment.invoke({
                        "desired_date": datetime_str,
                        "id_number": state['id_number'],
                        "doctor_name": extracted_doctor
                    })
                    final_message = direct_result
                except Exception as fallback_error:
                    final_message = f"Booking failed: {str(fallback_error)}"
            else:
                final_message = f"I encountered an error and could not extract booking information: {error_msg}"
        
        return Command(
            update={
                "messages": [AIMessage(content=final_message, name="booking_node")]
            },
            goto="supervisor",
        )

    def workflow(self):
        self.graph = StateGraph(AgentState)
        self.graph.add_node("supervisor", self.supervisor_node)
        self.graph.add_node("information_node", self.information_node)
        self.graph.add_node("booking_node", self.booking_node)
        self.graph.add_edge(START, "supervisor")
        self.app = self.graph.compile()
        return self.app