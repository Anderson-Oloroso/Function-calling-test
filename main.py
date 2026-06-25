import pandas as pd
from mistralai.client import Mistral
import json


data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}

df = pd.DataFrame(data)

def retrieve_payment_status(transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.values[0]})
    return json.dumps({'error': 'ID de la transaccion no fue encontrado'})

def retrieve_payment_date(transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.values[0]})
    return json.dumps({'error': 'ID de la transaccion no fue encontrado'})

name_to_functions = {
    'retrieve_payment_status': retrieve_payment_status,
    'retrieve_payment_date': retrieve_payment_date
}

messages = [
    {
        'role': 'system',
        'content': "Eres un asistente muy útil. Puedes usar las siguientes herramientas para ayudar a responder las preguntas de usuario relacionadas con los pagos de transacciones"
    }
]

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'retrieve_payment_status',
            'description': 'Obtener el estado de pago de una transacción',
            'parameters': {
                'type': 'object',
                'properties': {
                    'transaction_id': {
                        'type': 'string',
                        'description': 'id de la transaccion'
                    }
                },
                'required': ['transaction_id']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'retrieve_payment_date',
            'description': 'Obtener la fecha de pago de una transacción',
            'parameters': {  
                'type': 'object',
                'properties': {
                    'transaction_id': {
                        'type': 'string',
                        'description': 'id de la transaccion'
                    }
                },
                'required': ['transaction_id']
            }
        }
    }
]

api_key = 'SiCnrRy16FWKyzTFYtOriLi5qanGOj8B'
model = 'mistral-large-latest'
client = Mistral(api_key=api_key)
temperature = 0.1
top_p = 0.9

while True:
    user_query = input("Usuario: ")
    messages.append({'role': 'user', 'content': user_query})
    response = client.chat.complete(
        model=model,
        messages=messages,  
        tools=tools,
        temperature=temperature,
        top_p=top_p
    )

    messages.append(response.choices[0].message)

    while response.choices[0].message.tool_calls:
        if response.choices[0].message.content:
            print("--> Asistente: ", response.choices[0].message.content)

            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_params = json.loads(tool_call.function.arguments)
                function_result = name_to_functions[function_name](**function_params)

                print(f"Tool {tool_call.id}: ", f"{function_name} ({function_params} -> {function_result})")
                
                messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": function_result, 
                        "tool_call_id": tool_call.id
                    }
                )
            
            response = client.chat.complete(
                model=model,
                messages=messages, 
                tools=tools,
                temperature=temperature,
                top_p=top_p
            )

        messages.append(response.choices[0].message)

    print("--> Asistente: ", response.choices[0].message.content)