import pandas as pd
from typing import Literal
from langchain_core.tools import tool
from pydantic import ValidationError
from data_models.models import DateModel, DateTimeModel, IdentificationNumberModel


@tool
def check_availability_by_doctor(desired_date: str, doctor_name: str):
    """
    Checking the database if we have availability for the specific doctor.
    
    Args:
        desired_date: Date in format DD-MM-YYYY (e.g., 08-08-2024)
        doctor_name: Name of the doctor (lowercase with spaces, e.g., "emily johnson")
    
    Valid doctor names: kevin anderson, robert martinez, susan davis, daniel miller, 
    sarah wilson, michael green, lisa brown, jane smith, emily johnson, john doe
    """
    # Normalize doctor name to lowercase
    doctor_name = doctor_name.lower().strip()
    
    # Validate doctor name
    valid_doctors = ['kevin anderson', 'robert martinez', 'susan davis', 'daniel miller',
                     'sarah wilson', 'michael green', 'lisa brown', 'jane smith', 
                     'emily johnson', 'john doe']
    
    if doctor_name not in valid_doctors:
        return f"Invalid doctor name: '{doctor_name}'. Valid doctors are: {', '.join(valid_doctors)}"
    
    # Validate the date format
    try:
        validated_date = DateModel(date=desired_date)
    except ValidationError as e:
        return f"Invalid date format: {str(e)}. Please use DD-MM-YYYY format (e.g., 08-08-2024)"
    
    try:
        df = pd.read_csv(r"data/doctor_availability.csv")
        
        # Extract date from date_slot (format: DD-MM-YYYY HH:MM)
        df['date_only'] = df['date_slot'].str.split(' ').str[0]
        df['time_only'] = df['date_slot'].str.split(' ').str[1]
        
        rows = df[(df['date_only'] == validated_date.date) & 
                  (df['doctor_name'] == doctor_name) & 
                  (df['is_available'] == True)]

        if len(rows) == 0:
            # Check if doctor exists on that date but is booked
            all_slots = df[(df['date_only'] == validated_date.date) & 
                          (df['doctor_name'] == doctor_name)]
            if len(all_slots) > 0:
                return f"Dr. {doctor_name.title()} has no available slots on {validated_date.date}. All slots are booked."
            else:
                return f"Dr. {doctor_name.title()} is not available on {validated_date.date}."
        else:
            available_times = rows['time_only'].tolist()
            output = f'Availability for Dr. {doctor_name.title()} on {validated_date.date}:\n'
            output += "Available slots: " + ', '.join(available_times)
            return output
            
    except FileNotFoundError:
        return "Error: data/doctor_availability.csv file not found"
    except Exception as e:
        return f"Error checking availability: {str(e)}"


@tool
def check_availability_by_specialization(desired_date: str, specialization: Literal["general_dentist", "cosmetic_dentist", "prosthodontist", "pediatric_dentist","emergency_dentist","oral_surgeon","orthodontist"]):
    """
    Checking the database if we have availability for the specific specialization.
    Returns all available doctors with their time slots for that specialization.
    
    Args:
        desired_date: Date in format DD-MM-YYYY (e.g., 08-08-2024)
        specialization: Type of dental specialization
    """
    # Validate the date format
    try:
        validated_date = DateModel(date=desired_date)
    except ValidationError as e:
        return f"Invalid date format: {str(e)}. Please use DD-MM-YYYY format (e.g., 08-08-2024)"
    
    try:
        df = pd.read_csv(r"data\doctor_availability.csv")
        
        # Extract date and time from date_slot
        df['date_only'] = df['date_slot'].str.split(' ').str[0]
        df['time_only'] = df['date_slot'].str.split(' ').str[1]
        
        filtered_df = df[(df['date_only'] == validated_date.date) & 
                        (df['specialization'] == specialization) & 
                        (df['is_available'] == True)]
        
        if len(filtered_df) == 0:
            # Check if any slots exist for that specialization on that date
            all_slots = df[(df['date_only'] == validated_date.date) & 
                          (df['specialization'] == specialization)]
            if len(all_slots) > 0:
                return f"No available slots for {specialization.replace('_', ' ')} on {validated_date.date}. All slots are booked."
            else:
                return f"No {specialization.replace('_', ' ')} appointments available on {validated_date.date}. Please try another date."
        else:
            # Group by doctor and collect their available time slots
            rows = filtered_df.groupby('doctor_name')['time_only'].apply(list).reset_index()
            
            def convert_to_am_pm(time_str):
                time_str = str(time_str)
                hours, minutes = map(int, time_str.split(":"))
                period = "AM" if hours < 12 else "PM"
                hours = hours % 12 or 12
                return f"{hours}:{minutes:02d} {period}"
            
            output = f'Available {specialization.replace("_", " ")} appointments on {validated_date.date}:\n\n'
            for _, row in rows.iterrows():
                doctor_name = row['doctor_name'].title()
                slots = ', '.join([convert_to_am_pm(time) for time in row['time_only']])
                output += f"Dr. {doctor_name}:\n{slots}\n\n"
            
            return output.strip()
            
    except FileNotFoundError:
        return "Error: doctor_availability.csv file not found"
    except Exception as e:
        return f"Error checking availability: {str(e)}"


@tool
def set_appointment(desired_date: str, id_number: int, doctor_name: Literal['kevin anderson','robert martinez','susan davis','daniel miller','sarah wilson','michael green','lisa brown','jane smith','emily johnson','john doe']):
    """
    Set appointment or slot with the doctor.
    IMPORTANT: Only use this tool AFTER confirming availability with check_availability tools.
    
    Args:
        desired_date: DateTime in format DD-MM-YYYY HH:MM (e.g., 08-08-2024 20:00)
        id_number: Patient identification number (7 or 8 digits)
        doctor_name: Name of the doctor (must be exact match from the list)
    """
    # Validate inputs
    try:
        validated_date = DateTimeModel(date=desired_date)
        validated_id = IdentificationNumberModel(id=id_number)
    except ValidationError as e:
        return f"Invalid input: {str(e)}"
    
    try:
        df = pd.read_csv(r"data\doctor_availability.csv")
        
        # The date_slot in CSV is already in DD-MM-YYYY HH:MM format
        case = df[(df['date_slot'] == validated_date.date) & 
                  (df['doctor_name'] == doctor_name) & 
                  (df['is_available'] == True)]
        
        if len(case) == 0:
            # Check why it's not available
            slot_exists = df[(df['date_slot'] == validated_date.date) & (df['doctor_name'] == doctor_name)]
            if len(slot_exists) == 0:
                return f"No appointment slot exists for Dr. {doctor_name.title()} at {validated_date.date}. Please check availability first."
            else:
                return f"The slot at {validated_date.date} with Dr. {doctor_name.title()} is already booked."
        else:
            df.loc[(df['date_slot'] == validated_date.date) & 
                   (df['doctor_name'] == doctor_name) & 
                   (df['is_available'] == True), ['is_available','patient_to_attend']] = [False, validated_id.id]
            df.to_csv('data\doctor_availability.csv', index=False)

            return f"✓ Appointment successfully booked!\nDoctor: Dr. {doctor_name.title()}\nDate & Time: {validated_date.date}\nPatient ID: {validated_id.id}"
            
    except FileNotFoundError:
        return "Error: doctor_availability.csv file not found"
    except Exception as e:
        return f"Error setting appointment: {str(e)}"


@tool
def cancel_appointment(date: str, id_number: int, doctor_name: Literal['kevin anderson','robert martinez','susan davis','daniel miller','sarah wilson','michael green','lisa brown','jane smith','emily johnson','john doe']):
    """
    Canceling an appointment.
    
    Args:
        date: DateTime in format DD-MM-YYYY HH:MM (e.g., 08-08-2024 10:30)
        id_number: Patient identification number (7 or 8 digits)
        doctor_name: Name of the doctor
    """
    # Validate inputs
    try:
        validated_date = DateTimeModel(date=date)
        validated_id = IdentificationNumberModel(id=id_number)
    except ValidationError as e:
        return f"Invalid input: {str(e)}"
    
    try:
        df = pd.read_csv(r"data\doctor_availability.csv")
        
        # Convert patient_to_attend to int for comparison (it might be stored as float with NaN)
        df['patient_to_attend'] = pd.to_numeric(df['patient_to_attend'], errors='coerce')
        
        case_to_remove = df[(df['date_slot'] == validated_date.date) & 
                            (df['patient_to_attend'] == validated_id.id) & 
                            (df['doctor_name'] == doctor_name)]
        
        if len(case_to_remove) == 0:
            return f"No appointment found for patient ID {validated_id.id} with Dr. {doctor_name.title()} on {validated_date.date}"
        else:
            df.loc[(df['date_slot'] == validated_date.date) & 
                   (df['patient_to_attend'] == validated_id.id) & 
                   (df['doctor_name'] == doctor_name), ['is_available', 'patient_to_attend']] = [True, None]
            df.to_csv('data\doctor_availability.csv', index=False)

            return f"✓ Appointment successfully cancelled for patient ID {validated_id.id} with Dr. {doctor_name.title()} on {validated_date.date}"
            
    except FileNotFoundError:
        return "Error: doctor_availability.csv file not found"
    except Exception as e:
        return f"Error cancelling appointment: {str(e)}"


@tool
def reschedule_appointment(old_date: str, new_date: str, id_number: int, doctor_name: Literal['kevin anderson','robert martinez','susan davis','daniel miller','sarah wilson','michael green','lisa brown','jane smith','emily johnson','john doe']):
    """
    Rescheduling an appointment.
    
    Args:
        old_date: Old datetime in format DD-MM-YYYY HH:MM (e.g., 08-08-2024 10:30)
        new_date: New datetime in format DD-MM-YYYY HH:MM (e.g., 09-08-2024 14:00)
        id_number: Patient identification number (7 or 8 digits)
        doctor_name: Name of the doctor
    """
    # Validate inputs
    try:
        validated_old_date = DateTimeModel(date=old_date)
        validated_new_date = DateTimeModel(date=new_date)
        validated_id = IdentificationNumberModel(id=id_number)
    except ValidationError as e:
        return f"Invalid input: {str(e)}"
    
    try:
        df = pd.read_csv(r"data\doctor_availability.csv")
        available_for_desired_date = df[(df['date_slot'] == validated_new_date.date) & 
                                        (df['is_available'] == True) & 
                                        (df['doctor_name'] == doctor_name)]
        
        if len(available_for_desired_date) == 0:
            return f"Dr. {doctor_name.title()} is not available at {validated_new_date.date}. Please choose another time slot."
        else:
            cancel_result = cancel_appointment.invoke({
                'date': validated_old_date.date, 
                'id_number': validated_id.id, 
                'doctor_name': doctor_name
            })
            
            if "No appointment found" in cancel_result:
                return cancel_result
            
            set_result = set_appointment.invoke({
                'desired_date': validated_new_date.date, 
                'id_number': validated_id.id, 
                'doctor_name': doctor_name
            })
            
            if "successfully booked" in set_result:
                return f"✓ Appointment successfully rescheduled from {validated_old_date.date} to {validated_new_date.date} with Dr. {doctor_name.title()}"
            else:
                return set_result
                
    except FileNotFoundError:
        return "Error: doctor_availability.csv file not found"
    except Exception as e:
        return f"Error rescheduling appointment: {str(e)}"