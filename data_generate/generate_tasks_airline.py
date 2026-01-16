import json
import random
from datetime import datetime, timedelta

# Load existing data
with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\tasks.json', 'r', encoding='utf-8') as f:
    existing_tasks = json.load(f)

# Available tools (extracted from existing tasks)
TOOLS = [
    'book_reservation',
    'calculate',
    'cancel_reservation',
    'get_reservation_details',
    'get_user_details',
    'search_direct_flight',
    'send_certificate',
    'transfer_to_human_agents',
    'update_reservation_baggages',
    'update_reservation_flights',
    'update_reservation_passengers'
]

# Extract data from db
users = list(db['users'].values())
flights = list(db['flights'].values())
reservations = list(db['reservations'].values())

# Helper functions
def get_user_by_id(user_id):
    return db['users'].get(user_id)

def get_reservation_by_id(res_id):
    return db['reservations'].get(res_id)

def get_available_flights(origin=None, destination=None):
    """Get flights with available status on future dates"""
    available = []
    for flight in flights:
        if origin and flight['origin'] != origin:
            continue
        if destination and flight['destination'] != destination:
            continue
        for date, info in flight['dates'].items():
            if info['status'] == 'available':
                available.append({
                    'flight_number': flight['flight_number'],
                    'origin': flight['origin'],
                    'destination': flight['destination'],
                    'date': date,
                    'info': info
                })
    return available

# Task generation templates
def generate_book_flight_task(task_id):
    """Generate a task for booking a new flight"""
    user = random.choice(users)
    available_flights = get_available_flights()
    if not available_flights:
        return None

    flight = random.choice(available_flights)
    cabin = random.choice(['basic_economy', 'economy', 'business'])
    num_passengers = random.randint(1, 5)

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to book a {cabin} flight from {flight['origin']} to {flight['destination']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You want to book a {cabin} class flight. You have {num_passengers} passenger(s). Provide passenger details when asked.",
                "domain": "airline",
                "reason_for_call": f"You want to book a flight from {flight['origin']} to {flight['destination']} on {flight['date']}.",
                "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.",
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": f"{task_id}_0",
                    "name": "get_user_details",
                    "arguments": {"user_id": user['user_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_1",
                    "name": "search_direct_flight",
                    "arguments": {
                        "origin": flight['origin'],
                        "destination": flight['destination']
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should search for flights from {flight['origin']} to {flight['destination']}.",
                f"Agent should attempt to book a {cabin} class flight."
            ]
        },
        "annotations": None
    }
    return task

def generate_cancel_reservation_task(task_id):
    """Generate a task for canceling a reservation"""
    # Find reservations that can be cancelled
    valid_reservations = []
    for res in reservations:
        # Check if reservation is in the future
        if res['flights']:
            first_flight_date = res['flights'][0]['date']
            if first_flight_date >= '2024-05-15':  # Current time in policy
                valid_reservations.append(res)

    if not valid_reservations:
        return None

    reservation = random.choice(valid_reservations)
    user = get_user_by_id(reservation['user_id'])
    if not user:
        return None

    # Determine if cancellation should be allowed
    cabin = reservation['cabin']
    has_insurance = reservation.get('insurance', 'no') == 'yes'
    is_business = cabin == 'business'

    # Calculate if within 24 hours
    created_at = datetime.fromisoformat(reservation['created_at'].replace('T', ' '))
    current_time = datetime(2024, 5, 15, 15, 0, 0)
    within_24h = (current_time - created_at).total_seconds() < 24 * 3600

    can_cancel = within_24h or is_business or has_insurance
    reason = random.choice(['change of plan', 'sick', 'emergency'])

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to cancel reservation {reservation['reservation_id']}. {'Should be allowed' if can_cancel else 'Should be denied'}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You want to cancel your reservation. The reason is {reason}.",
                "domain": "airline",
                "reason_for_call": f"You want to cancel your reservation {reservation['reservation_id']} and get a refund.",
                "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.\nYour reservation number is {reservation['reservation_id']}.",
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": f"{task_id}_0",
                    "name": "get_user_details",
                    "arguments": {"user_id": user['user_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservation['reservation_id']},
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should {'cancel' if can_cancel else 'not cancel'} reservation {reservation['reservation_id']}."
            ]
        },
        "annotations": None
    }

    if can_cancel:
        task["evaluation_criteria"]["actions"].append({
            "action_id": f"{task_id}_2",
            "name": "cancel_reservation",
            "arguments": {"reservation_id": reservation['reservation_id']},
            "info": None
        })

    return task

def generate_modify_flight_task(task_id):
    """Generate a task for modifying a flight reservation"""
    # Find modifiable reservations (not basic economy, future flights)
    valid_reservations = []
    for res in reservations:
        if res['cabin'] != 'basic_economy' and res['flights']:
            first_flight_date = res['flights'][0]['date']
            if first_flight_date >= '2024-05-15':
                valid_reservations.append(res)

    if not valid_reservations:
        return None

    reservation = random.choice(valid_reservations)
    user = get_user_by_id(reservation['user_id'])
    if not user:
        return None

    # Find alternative flights
    origin = reservation['origin']
    destination = reservation['destination']
    available = get_available_flights(origin, destination)

    if not available:
        return None

    new_flight = random.choice(available)

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to change flight in reservation {reservation['reservation_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You want to change your flight to {new_flight['date']}. Confirm the change if there are additional costs.",
                "domain": "airline",
                "reason_for_call": f"You need to change the date of your flight from {origin} to {destination}.",
                "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.\nYour reservation number is {reservation['reservation_id']}.",
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": f"{task_id}_0",
                    "name": "get_user_details",
                    "arguments": {"user_id": user['user_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservation['reservation_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_2",
                    "name": "search_direct_flight",
                    "arguments": {
                        "origin": origin,
                        "destination": destination
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should search for alternative flights from {origin} to {destination}.",
                f"Agent should attempt to modify reservation {reservation['reservation_id']}."
            ]
        },
        "annotations": None
    }

    return task

def generate_add_baggage_task(task_id):
    """Generate a task for adding baggage to a reservation"""
    # Find reservations with future flights
    valid_reservations = []
    for res in reservations:
        if res['flights']:
            first_flight_date = res['flights'][0]['date']
            if first_flight_date >= '2024-05-15':
                valid_reservations.append(res)

    if not valid_reservations:
        return None

    reservation = random.choice(valid_reservations)
    user = get_user_by_id(reservation['user_id'])
    if not user:
        return None

    num_bags = random.randint(1, 3)

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to add {num_bags} checked bag(s) to reservation {reservation['reservation_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You need to add {num_bags} checked bag(s) to your reservation. Confirm if there are additional fees.",
                "domain": "airline",
                "reason_for_call": f"You want to add checked baggage to your reservation.",
                "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.\nYour reservation number is {reservation['reservation_id']}.",
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": f"{task_id}_0",
                    "name": "get_user_details",
                    "arguments": {"user_id": user['user_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservation['reservation_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_2",
                    "name": "update_reservation_baggages",
                    "arguments": {
                        "reservation_id": reservation['reservation_id'],
                        "new_total_baggages": reservation['total_baggages'] + num_bags
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should add {num_bags} checked bag(s) to reservation {reservation['reservation_id']}."
            ]
        },
        "annotations": None
    }

    return task

def generate_change_cabin_task(task_id):
    """Generate a task for changing cabin class"""
    # Find reservations with future flights that haven't been flown
    valid_reservations = []
    for res in reservations:
        if res['flights']:
            first_flight_date = res['flights'][0]['date']
            if first_flight_date >= '2024-05-15':
                valid_reservations.append(res)

    if not valid_reservations:
        return None

    reservation = random.choice(valid_reservations)
    user = get_user_by_id(reservation['user_id'])
    if not user:
        return None

    current_cabin = reservation['cabin']
    cabins = ['basic_economy', 'economy', 'business']
    cabins.remove(current_cabin)
    new_cabin = random.choice(cabins)

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to change cabin from {current_cabin} to {new_cabin} for reservation {reservation['reservation_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You want to upgrade/change your cabin to {new_cabin}. Accept the price difference if needed.",
                "domain": "airline",
                "reason_for_call": f"You want to change your cabin class to {new_cabin}.",
                "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.\nYour reservation number is {reservation['reservation_id']}.",
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": f"{task_id}_0",
                    "name": "get_user_details",
                    "arguments": {"user_id": user['user_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservation['reservation_id']},
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should attempt to change cabin from {current_cabin} to {new_cabin}."
            ]
        },
        "annotations": None
    }

    return task

def generate_compensation_task(task_id):
    """Generate a task for requesting compensation for delayed/cancelled flights"""
    # Find reservations with cancelled or delayed flights
    valid_reservations = []
    for res in reservations:
        if res['flights']:
            for flight_info in res['flights']:
                flight_num = flight_info['flight_number']
                date = flight_info['date']
                # Find the flight in db
                for f in flights:
                    if f['flight_number'] == flight_num:
                        if date in f['dates']:
                            status = f['dates'][date]['status']
                            if status in ['cancelled', 'delayed']:
                                valid_reservations.append((res, status))
                                break

    if not valid_reservations:
        return None

    reservation, status = random.choice(valid_reservations)
    user = get_user_by_id(reservation['user_id'])
    if not user:
        return None

    # Check if eligible for compensation
    membership = user.get('membership', 'regular')
    has_insurance = reservation.get('insurance', 'no') == 'yes'
    cabin = reservation['cabin']
    is_business = cabin == 'business'

    eligible = (membership in ['silver', 'gold']) or has_insurance or is_business

    if not eligible:
        return None

    num_passengers = len(reservation['passengers'])
    amount = 100 * num_passengers if status == 'cancelled' else 50 * num_passengers

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User complains about {status} flight and requests compensation.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"Complain about the {status} flight. Ask for compensation if agent doesn't offer it.",
                "domain": "airline",
                "reason_for_call": f"Your flight was {status} and you want compensation.",
                "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.\nYour reservation number is {reservation['reservation_id']}.",
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": f"{task_id}_0",
                    "name": "get_user_details",
                    "arguments": {"user_id": user['user_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservation['reservation_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_2",
                    "name": "send_certificate",
                    "arguments": {
                        "user_id": user['user_id'],
                        "amount": amount
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should verify the flight was {status}.",
                f"Agent should offer a certificate of ${amount}."
            ]
        },
        "annotations": None
    }

    return task

def generate_update_passenger_task(task_id):
    """Generate a task for updating passenger information"""
    # Find reservations with future flights
    valid_reservations = []
    for res in reservations:
        if res['flights'] and res['passengers']:
            first_flight_date = res['flights'][0]['date']
            if first_flight_date >= '2024-05-15':
                valid_reservations.append(res)

    if not valid_reservations:
        return None

    reservation = random.choice(valid_reservations)
    user = get_user_by_id(reservation['user_id'])
    if not user:
        return None

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to update passenger information for reservation {reservation['reservation_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": "You need to correct a passenger's name due to a typo. Provide the corrected information when asked.",
                "domain": "airline",
                "reason_for_call": "You need to update passenger information in your reservation.",
                "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.\nYour reservation number is {reservation['reservation_id']}.",
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": f"{task_id}_0",
                    "name": "get_user_details",
                    "arguments": {"user_id": user['user_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": reservation['reservation_id']},
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should attempt to update passenger information for reservation {reservation['reservation_id']}."
            ]
        },
        "annotations": None
    }

    return task

# Generate tasks
def generate_all_tasks(start_id, end_id):
    """Generate tasks from start_id to end_id (inclusive)"""
    tasks = []
    task_generators = [
        generate_book_flight_task,
        generate_cancel_reservation_task,
        generate_modify_flight_task,
        generate_add_baggage_task,
        generate_change_cabin_task,
        generate_compensation_task,
        generate_update_passenger_task
    ]

    for task_id in range(start_id, end_id + 1):
        # Randomly select a task type
        generator = random.choice(task_generators)
        task = None
        attempts = 0

        # Try to generate a valid task (max 10 attempts)
        while task is None and attempts < 10:
            task = generator(task_id)
            attempts += 1

        # If still None, try other generators
        if task is None:
            for gen in task_generators:
                task = gen(task_id)
                if task is not None:
                    break

        if task is not None:
            tasks.append(task)
            print(f"Generated task {task_id}")
        else:
            print(f"Failed to generate task {task_id}")

    return tasks

# Main execution
if __name__ == "__main__":
    print("Generating tasks from ID 50 to 299...")
    new_tasks = generate_all_tasks(50, 299)

    # Combine with existing tasks
    all_tasks = existing_tasks + new_tasks

    # Save to file
    output_file = r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\tasks.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)

    print(f"\nTotal tasks generated: {len(new_tasks)}")
    print(f"Total tasks in file: {len(all_tasks)}")
    print(f"Saved to {output_file}")
