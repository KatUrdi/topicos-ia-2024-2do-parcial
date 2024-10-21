from fastapi import FastAPI, Depends, Query
from llama_index.core.agent import ReActAgent
from ai_assistant.agent import TravelAgent
from ai_assistant.config import get_agent_settings
from ai_assistant.models import (
    AgentAPIResponse, 
    RecommendationRequest, 
    ReservationRequest, 
    HotelReservationRequest, 
    RestaurantReservationRequest, 
    TripReservation
)
from ai_assistant.tools import (
    reserve_flight, 
    reserve_bus, 
    reserve_hotel, 
    reserve_restaurant
)
from ai_assistant.prompts import agent_prompt_tpl

SETTINGS = get_agent_settings()

def get_agent() -> ReActAgent:
    return TravelAgent(agent_prompt_tpl).get_agent()

app = FastAPI(title="Tourism AI Assistant")

@app.get("/recommendations/cities")
def recommend_cities(
    preferences: list[str] = Query(...), agent: ReActAgent = Depends(get_agent)
):
    prompt = f"Suggest Bolivian cities to visit based on the following preferences: {preferences}"
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))

@app.get("/recommendations/places")
def recommend_places(
    city: str = Query(...),
    preferences: list[str] = Query(None), agent: ReActAgent = Depends(get_agent)
):
    if preferences:
        prompt = f"Suggest places to explore in {city}, Bolivia, considering these preferences: {preferences}. Focus only on tourist attractions, excluding restaurants, hotels, or activities."
    else:
        prompt = f"Suggest places to explore in {city}, Bolivia, excluding restaurants, hotels, or activities."
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))

@app.get("/recommendations/hotels")
def recommend_hotels(
    city: str = Query(...),
    preferences: list[str] = Query(None), agent: ReActAgent = Depends(get_agent)
):
    if preferences:
        prompt = f"Recommend hotels in {city}, Bolivia, based on these preferences: {preferences}"
    else:
        prompt = f"Recommend hotels to stay in {city}, Bolivia."
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))

@app.get("/recommendations/activities")
def recommend_activities(
    city: str = Query(...),
    preferences: list[str] = Query(None), agent: ReActAgent = Depends(get_agent)
):
    if preferences:
        prompt = f"Suggest activities to do in {city}, Bolivia, taking into account these preferences: {preferences}"
    else:
        prompt = f"Suggest activities to do in {city}, Bolivia."
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))

@app.post("/reservations/flight")
def reserve_flight_api(request: ReservationRequest = Query(...)):
    reservation = reserve_flight(
        destination=request.destination,
        origin=request.origin,
        date=request.date
    )
    return AgentAPIResponse(status="OK", agent_response=str(reservation))

@app.post("/reservations/bus")
def reserve_bus_api(request: ReservationRequest = Query(...)):
    reservation = reserve_bus(
        date=request.date,
        origin=request.origin,
        destination=request.destination
    )
    return AgentAPIResponse(status="OK", agent_response=str(reservation))

@app.post("/reservations/hotel")
def reserve_hotel_api(request: HotelReservationRequest = Query(...)):
    reservation = reserve_hotel(
        checkin_date=request.checkin_date,
        checkout_date=request.checkout_date,
        hotel=request.hotel, 
        city=request.city
    )
    return AgentAPIResponse(status="OK", agent_response=str(reservation))

@app.post("/reservations/restaurant")
def reserve_restaurant_api(request: RestaurantReservationRequest = Query(...)):
    dish = request.dish or "unspecified"
    reservation = reserve_restaurant(
        f"{request.date}T{request.time}", 
        request.restaurant, 
        request.city, 
        dish
    )
    return AgentAPIResponse(status="OK", agent_response=str(reservation))

@app.get("/trip_summary")
def trip_summary(agent: ReActAgent = Depends(get_agent)):
    prompt = """
    Generate a trip summary using the `trip_summary_tool`. Afterward, provide a detailed report that includes:
    1. Main highlights of the trip.
    2. Potential issues or recommendations.
    3. Additional insights based on the summary.

    Please ensure the summary and the detailed report are in Spanish and provide them both as the final response.
    """
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))
