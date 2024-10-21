import json
from random import randint
from datetime import date, datetime
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata
from ai_assistant.rags import TravelGuideRAG
from ai_assistant.prompts import travel_guide_qa_tpl, travel_guide_description
from ai_assistant.config import get_agent_settings
from ai_assistant.models import (
    TripReservation,
    TripType,
    HotelReservation,
    RestaurantReservation,
)
from ai_assistant.utils import save_reservation



SETTINGS = get_agent_settings()

travel_guide_tool = QueryEngineTool(
    query_engine=TravelGuideRAG(
        store_path=SETTINGS.travel_guide_store_path,
        data_dir=SETTINGS.travel_guide_data_path,
        qa_prompt_tpl=travel_guide_qa_tpl,
    ).get_query_engine(),
    metadata=ToolMetadata(
        name="travel_guide",
        description=travel_guide_description,
        return_direct=False,
    ),
)

def reserve_flight(destination: str, origin: str, date_str: str) -> TripReservation:
    """
    Books a flight for the user by specifying the destination, origin, and flight date.

    ### Inputs:
    1. **destination**: Target city.
    2. **origin**: Departure city.
    3. **date_str**: Flight date (format: 'YYYY-MM-DD').

    ### Output:
    - Returns the flight reservation details:
        - Cities (origin and destination)
        - Flight date
        - Cost
    
    ### Notes:
    - The reservation is saved for future reference.
    - Use when the user requests to book a flight.
    """
    print(f"Reserving flight from {origin} to {destination} on {date_str}")
    reservation = TripReservation(
        trip_type=TripType.flight,
        departure=origin,
        destination=destination,
        date=date.fromisoformat(date_str),
        cost=randint(200, 700),
    )

    save_reservation(reservation)
    return reservation

flight_tool = FunctionTool.from_defaults(fn=reserve_flight, return_direct=False)

def reserve_bus(date_str: str, origin: str, destination: str) -> TripReservation:
    """
    Books a bus trip for the user by specifying the trip date, origin, and destination.

    ### Inputs:
    1. **date_str**: Date of the bus trip (format: 'YYYY-MM-DD').
    2. **origin**: Departure city.
    3. **destination**: Destination city.

    ### Output:
    - Returns the bus reservation details:
        - Cities (origin and destination)
        - Trip date
        - Cost
    
    ### Notes:
    - The reservation is saved for future reference.
    - Use when the user requests to book a bus trip.
    """
    print(f"Reserving bus from {origin} to {destination} on {date_str}")
    reservation = TripReservation(
        trip_type=TripType.bus,
        departure=origin,
        destination=destination,
        date=date.fromisoformat(date_str),
        cost=randint(50, 350),
    )

    save_reservation(reservation)
    return reservation

bus_tool = FunctionTool.from_defaults(fn=reserve_bus, return_direct=False)

def reserve_hotel(checkin_str: str, checkout_str: str, hotel_name: str, city: str) -> HotelReservation:
    """
    Reserves a hotel stay by specifying the check-in and check-out dates, hotel name, and location.

    ### Inputs:
    1. **checkin_str**: Check-in date (format: 'YYYY-MM-DD').
    2. **checkout_str**: Check-out date (format: 'YYYY-MM-DD').
    3. **hotel_name**: Hotel name.
    4. **city**: City where the hotel is located.

    ### Output:
    - Returns the hotel reservation details:
        - Check-in and check-out dates
        - Hotel name
        - City
        - Cost
    
    ### Notes:
    - The reservation is saved for future reference.
    - Use when the user requests to book a hotel stay.
    """
    print(f"Reserving hotel at {hotel_name} in {city} from {checkin_str} to {checkout_str}")
    reservation = HotelReservation(
        checkin_date=date.fromisoformat(checkin_str),
        checkout_date=date.fromisoformat(checkout_str),
        hotel_name=hotel_name,
        city=city,
        cost=randint(500, 1000),
    )

    save_reservation(reservation)
    return reservation

hotel_tool = FunctionTool.from_defaults(fn=reserve_hotel, return_direct=False)

def reserve_restaurant(reservation_time_str: str, restaurant: str, city: str, dish: str = "not specified") -> RestaurantReservation:
    """
    Reserves a table at a restaurant by specifying the date and time, restaurant name, and location.

    ### Inputs:
    1. **reservation_time_str**: Reservation time (format: 'YYYY-MM-DDTHH:MM:SS').
    2. **restaurant**: Restaurant name.
    3. **city**: City where the restaurant is located.
    4. **dish**: (Optional) Dish the user wants to order.

    ### Output:
    - Returns the restaurant reservation details:
        - Reservation time
        - Restaurant name
        - City
        - Cost
    
    ### Notes:
    - The reservation is saved for future reference.
    - Use when the user requests to book a restaurant.
    """
    reservation_time = datetime.fromisoformat(reservation_time_str)
    print(f"Reserving table at {restaurant} in {city} at {reservation_time}")
    reservation = RestaurantReservation(
        reservation_time=reservation_time,
        restaurant=restaurant,
        city=city,
        dish=dish,
        cost=randint(100, 500),
    )

    save_reservation(reservation)
    return reservation

restaurant_tool = FunctionTool.from_defaults(fn=reserve_restaurant, return_direct=False)

def trip_summary() -> str:
    """
    Generates a summary of the user's trip based on stored reservations.

    ### Output:
    - Returns a trip summary including:
        - Activities by city and date
        - Total cost of the trip
    
    ### Notes:
    - The tool retrieves all trip data from the 'trip.json' file.
    - Use to provide the user with an overview of their trip plans and costs.
    """
    with open(SETTINGS.log_file, "r") as file:
        trip_data = json.load(file)

    activities_by_city = {}
    total_cost = 0

    for activity in trip_data:
        city = activity.get('city', activity.get('departure', 'unknown'))
        date = activity.get('date', activity.get('checkin_date', activity.get('reservation_time', 'unknown')))
        cost = activity.get('cost', 0)
        total_cost += cost

        if city not in activities_by_city:
            activities_by_city[city] = []

        activities_by_city[city].append({
            'activity': activity.get('reservation_type', 'Activity'),
            'date': date,
            'details': activity,
        })

    summary = "Trip Summary:\n\n"
    for city, activities in activities_by_city.items():
        summary += f"City: {city}\n"
        for activity in activities:
            summary += f"  - Activity: {activity['activity']}\n"
            summary += f"    Date: {activity['date']}\n"
            summary += f"    Details: {json.dumps(activity['details'], indent=2)}\n"
        summary += "\n"

    summary += f"Total Cost: ${total_cost:.2f}\n"

    return summary

trip_summary_tool = FunctionTool.from_defaults(fn=trip_summary, return_direct=False)

