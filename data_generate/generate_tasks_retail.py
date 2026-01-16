import json
import random

# Load existing data
with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\retail\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\retail\tasks.json', 'r', encoding='utf-8') as f:
    existing_tasks = json.load(f)

# Available tools (extracted from existing tasks)
TOOLS = [
    'calculate',
    'cancel_pending_order',
    'exchange_delivered_order_items',
    'find_user_id_by_email',
    'find_user_id_by_name_zip',
    'get_order_details',
    'get_product_details',
    'get_user_details',
    'modify_pending_order_address',
    'modify_pending_order_items',
    'modify_pending_order_payment',
    'modify_user_address',
    'return_delivered_order_items',
    'transfer_to_human_agents'
]

# Extract data from db
users = list(db['users'].values())
products = list(db['products'].values())
orders = list(db['orders'].values())

# Helper functions
def get_user_by_id(user_id):
    return db['users'].get(user_id)

def get_order_by_id(order_id):
    return db['orders'].get(order_id)

def get_product_by_id(product_id):
    return db['products'].get(product_id)

def get_orders_by_status(status):
    """Get orders with specific status"""
    return [o for o in orders if o['status'] == status]

def get_user_orders(user_id, status=None):
    """Get orders for a specific user"""
    user = get_user_by_id(user_id)
    if not user:
        return []

    user_orders = []
    for order_id in user.get('orders', []):
        order = get_order_by_id(order_id)
        if order:
            if status is None or order['status'] == status:
                user_orders.append(order)
    return user_orders

# Task generation templates
def generate_cancel_pending_order_task(task_id):
    """Generate a task for canceling a pending order"""
    pending_orders = get_orders_by_status('pending')
    if not pending_orders:
        return None

    order = random.choice(pending_orders)
    user = get_user_by_id(order['user_id'])
    if not user:
        return None

    reason = random.choice(['no longer needed', 'ordered by mistake'])
    auth_method = random.choice(['email', 'name_zip'])

    if auth_method == 'email':
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour email is {user['email']}."
        unknown_info = "You do not remember your zip code."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_email",
            "arguments": {"email": user['email']},
            "info": None
        }
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."
        unknown_info = "You do not remember your email address."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_name_zip",
            "arguments": {
                "first_name": user['name']['first_name'],
                "last_name": user['name']['last_name'],
                "zip": user['address']['zip']
            },
            "info": None
        }

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to cancel pending order {order['order_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You want to cancel your order because it is {reason}. Confirm the cancellation when asked.",
                "domain": "retail",
                "reason_for_call": f"You want to cancel your order {order['order_id']}.",
                "known_info": known_info,
                "unknown_info": unknown_info
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                auth_action,
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_order_details",
                    "arguments": {"order_id": order['order_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_2",
                    "name": "cancel_pending_order",
                    "arguments": {
                        "order_id": order['order_id'],
                        "reason": reason
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should cancel order {order['order_id']}."
            ]
        },
        "annotations": None
    }
    return task

def generate_return_delivered_order_task(task_id):
    """Generate a task for returning items from a delivered order"""
    delivered_orders = get_orders_by_status('delivered')
    if not delivered_orders:
        return None

    order = random.choice(delivered_orders)
    user = get_user_by_id(order['user_id'])
    if not user or not order['items']:
        return None

    # Select items to return (1-3 items)
    num_items_to_return = min(random.randint(1, 3), len(order['items']))
    items_to_return = random.sample(order['items'], num_items_to_return)

    # Select payment method for refund
    payment_methods = list(user['payment_methods'].values())
    refund_method = random.choice(payment_methods)

    auth_method = random.choice(['email', 'name_zip'])

    if auth_method == 'email':
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour email is {user['email']}."
        unknown_info = None
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_email",
            "arguments": {"email": user['email']},
            "info": None
        }
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."
        unknown_info = None
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_name_zip",
            "arguments": {
                "first_name": user['name']['first_name'],
                "last_name": user['name']['last_name'],
                "zip": user['address']['zip']
            },
            "info": None
        }

    items_desc = ', '.join([item['name'] for item in items_to_return])

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to return items from delivered order {order['order_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You want to return the following items: {items_desc}. Confirm the return when asked.",
                "domain": "retail",
                "reason_for_call": f"You received your order {order['order_id']} but want to return some items.",
                "known_info": known_info,
                "unknown_info": unknown_info
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                auth_action,
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_order_details",
                    "arguments": {"order_id": order['order_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_2",
                    "name": "return_delivered_order_items",
                    "arguments": {
                        "order_id": order['order_id'],
                        "item_ids": [item['item_id'] for item in items_to_return],
                        "payment_method_id": refund_method['id']
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should process return for order {order['order_id']}."
            ]
        },
        "annotations": None
    }
    return task

def generate_exchange_delivered_order_task(task_id):
    """Generate a task for exchanging items from a delivered order"""
    delivered_orders = get_orders_by_status('delivered')
    if not delivered_orders:
        return None

    order = random.choice(delivered_orders)
    user = get_user_by_id(order['user_id'])
    if not user or not order['items']:
        return None

    # Select an item to exchange
    item_to_exchange = random.choice(order['items'])
    product = get_product_by_id(item_to_exchange['product_id'])
    if not product or len(product['variants']) < 2:
        return None

    # Find a different variant of the same product
    available_variants = [v for v in product['variants'].values()
                         if v['available'] and v['item_id'] != item_to_exchange['item_id']]
    if not available_variants:
        return None

    new_variant = random.choice(available_variants)

    # Select payment method
    payment_methods = list(user['payment_methods'].values())
    payment_method = random.choice(payment_methods)

    auth_method = random.choice(['email', 'name_zip'])

    if auth_method == 'email':
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour email is {user['email']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_email",
            "arguments": {"email": user['email']},
            "info": None
        }
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_name_zip",
            "arguments": {
                "first_name": user['name']['first_name'],
                "last_name": user['name']['last_name'],
                "zip": user['address']['zip']
            },
            "info": None
        }

    # Describe the exchange
    old_options = ', '.join([f"{k}: {v}" for k, v in item_to_exchange['options'].items()])
    new_options = ', '.join([f"{k}: {v}" for k, v in new_variant['options'].items()])

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to exchange {item_to_exchange['name']} in order {order['order_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You want to exchange the {item_to_exchange['name']} ({old_options}) for one with different options ({new_options}). Confirm the exchange when asked.",
                "domain": "retail",
                "reason_for_call": f"You received your order {order['order_id']} but want to exchange an item.",
                "known_info": known_info,
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                auth_action,
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_order_details",
                    "arguments": {"order_id": order['order_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_2",
                    "name": "get_product_details",
                    "arguments": {"product_id": product['product_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_3",
                    "name": "exchange_delivered_order_items",
                    "arguments": {
                        "order_id": order['order_id'],
                        "old_item_ids": [item_to_exchange['item_id']],
                        "new_item_ids": [new_variant['item_id']],
                        "payment_method_id": payment_method['id']
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should process exchange for order {order['order_id']}."
            ]
        },
        "annotations": None
    }
    return task

def generate_modify_pending_order_address_task(task_id):
    """Generate a task for modifying address of a pending order"""
    pending_orders = get_orders_by_status('pending')
    if not pending_orders:
        return None

    order = random.choice(pending_orders)
    user = get_user_by_id(order['user_id'])
    if not user:
        return None

    # Generate a new address (use user's default address with slight modification)
    new_address = user['address'].copy()
    new_address['address1'] = f"{random.randint(100, 999)} {random.choice(['Main', 'Oak', 'Maple', 'River'])} Street"

    auth_method = random.choice(['email', 'name_zip'])

    if auth_method == 'email':
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour email is {user['email']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_email",
            "arguments": {"email": user['email']},
            "info": None
        }
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_name_zip",
            "arguments": {
                "first_name": user['name']['first_name'],
                "last_name": user['name']['last_name'],
                "zip": user['address']['zip']
            },
            "info": None
        }

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to change shipping address for pending order {order['order_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You need to change the shipping address to {new_address['address1']}, {new_address['city']}, {new_address['state']} {new_address['zip']}. Confirm when asked.",
                "domain": "retail",
                "reason_for_call": f"You want to change the shipping address for your pending order {order['order_id']}.",
                "known_info": known_info,
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                auth_action,
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_order_details",
                    "arguments": {"order_id": order['order_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_2",
                    "name": "modify_pending_order_address",
                    "arguments": {
                        "order_id": order['order_id'],
                        "new_address": new_address
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should modify shipping address for order {order['order_id']}."
            ]
        },
        "annotations": None
    }
    return task

def generate_modify_pending_order_payment_task(task_id):
    """Generate a task for modifying payment method of a pending order"""
    pending_orders = get_orders_by_status('pending')
    if not pending_orders:
        return None

    order = random.choice(pending_orders)
    user = get_user_by_id(order['user_id'])
    if not user or len(user['payment_methods']) < 2:
        return None

    # Get current payment method
    current_payment_id = order['payment_history'][0]['payment_method_id']

    # Select a different payment method
    available_methods = [pm for pm in user['payment_methods'].values()
                        if pm['id'] != current_payment_id]
    if not available_methods:
        return None

    new_payment_method = random.choice(available_methods)

    auth_method = random.choice(['email', 'name_zip'])

    if auth_method == 'email':
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour email is {user['email']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_email",
            "arguments": {"email": user['email']},
            "info": None
        }
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_name_zip",
            "arguments": {
                "first_name": user['name']['first_name'],
                "last_name": user['name']['last_name'],
                "zip": user['address']['zip']
            },
            "info": None
        }

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to change payment method for pending order {order['order_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You want to change the payment method to a different one. Confirm when asked.",
                "domain": "retail",
                "reason_for_call": f"You want to change the payment method for your pending order {order['order_id']}.",
                "known_info": known_info,
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                auth_action,
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_order_details",
                    "arguments": {"order_id": order['order_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_2",
                    "name": "get_user_details",
                    "arguments": {"user_id": user['user_id']},
                    "info": None
                },
                {
                    "action_id": f"{task_id}_3",
                    "name": "modify_pending_order_payment",
                    "arguments": {
                        "order_id": order['order_id'],
                        "new_payment_method_id": new_payment_method['id']
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should modify payment method for order {order['order_id']}."
            ]
        },
        "annotations": None
    }
    return task

def generate_modify_user_address_task(task_id):
    """Generate a task for modifying user's default address"""
    user = random.choice(users)

    # Generate a new address
    new_address = {
        "address1": f"{random.randint(100, 999)} {random.choice(['Main', 'Oak', 'Maple', 'River', 'Park'])} {random.choice(['Street', 'Avenue', 'Drive', 'Road'])}",
        "address2": f"Apt {random.randint(1, 999)}" if random.random() > 0.5 else "",
        "city": user['address']['city'],
        "country": "USA",
        "state": user['address']['state'],
        "zip": user['address']['zip']
    }

    auth_method = random.choice(['email', 'name_zip'])

    if auth_method == 'email':
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour email is {user['email']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_email",
            "arguments": {"email": user['email']},
            "info": None
        }
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_name_zip",
            "arguments": {
                "first_name": user['name']['first_name'],
                "last_name": user['name']['last_name'],
                "zip": user['address']['zip']
            },
            "info": None
        }

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to update their default address.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": f"You want to update your default address to {new_address['address1']}, {new_address['city']}, {new_address['state']} {new_address['zip']}. Confirm when asked.",
                "domain": "retail",
                "reason_for_call": "You want to update your default shipping address.",
                "known_info": known_info,
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                auth_action,
                {
                    "action_id": f"{task_id}_1",
                    "name": "modify_user_address",
                    "arguments": {
                        "user_id": user['user_id'],
                        "new_address": new_address
                    },
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should update default address for user {user['user_id']}."
            ]
        },
        "annotations": None
    }
    return task

def generate_get_order_info_task(task_id):
    """Generate a task for getting order information"""
    user = random.choice(users)
    if not user.get('orders'):
        return None

    order_id = random.choice(user['orders'])
    order = get_order_by_id(order_id)
    if not order:
        return None

    auth_method = random.choice(['email', 'name_zip'])

    if auth_method == 'email':
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour email is {user['email']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_email",
            "arguments": {"email": user['email']},
            "info": None
        }
    else:
        known_info = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."
        auth_action = {
            "action_id": f"{task_id}_0",
            "name": "find_user_id_by_name_zip",
            "arguments": {
                "first_name": user['name']['first_name'],
                "last_name": user['name']['last_name'],
                "zip": user['address']['zip']
            },
            "info": None
        }

    task = {
        "id": str(task_id),
        "description": {
            "purpose": f"User wants to check status of order {order['order_id']}.",
            "relevant_policies": None,
            "notes": None
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": "You want to know the status of your order and when it will arrive.",
                "domain": "retail",
                "reason_for_call": f"You want to check the status of your order {order['order_id']}.",
                "known_info": known_info,
                "unknown_info": None
            }
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                auth_action,
                {
                    "action_id": f"{task_id}_1",
                    "name": "get_order_details",
                    "arguments": {"order_id": order['order_id']},
                    "info": None
                }
            ],
            "communicate_info": [],
            "nl_assertions": [
                f"Agent should provide status information for order {order['order_id']}."
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
        generate_cancel_pending_order_task,
        generate_return_delivered_order_task,
        generate_exchange_delivered_order_task,
        generate_modify_pending_order_address_task,
        generate_modify_pending_order_payment_task,
        generate_modify_user_address_task,
        generate_get_order_info_task
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
    print("Generating retail tasks from ID 114 to 299...")
    new_tasks = generate_all_tasks(114, 299)

    # Combine with existing tasks
    all_tasks = existing_tasks + new_tasks

    # Save to file
    output_file = r'D:\Desktop\work\tau2-bench\data\tau2\domains\retail\tasks.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)

    print(f"\nTotal tasks generated: {len(new_tasks)}")
    print(f"Total tasks in file: {len(all_tasks)}")
    print(f"Saved to {output_file}")
