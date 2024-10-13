import diskcache
from dash.long_callback import DiskcacheLongCallbackManager
import json
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

K = 6
clf_model = AutoModelForSequenceClassification.from_pretrained(
    "Amirth24/disease_classifier")
tokenizer = AutoTokenizer.from_pretrained("Amirth24/disease_classifier")

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)


app.layout = [
    dbc.Container([
        html.H1("Tell about your symptoms"),
        dcc.Input(id="sym-input", type="text",
                  placeholder="Enter your symptoms"),
        dbc.Button(id="submit", children="Find out",
                   class_name="btn btn-primary"),
    ], class_name="mt-3 text-center "),
    html.Div(id="output"),
]


def predict_disease(prompt):
    tokenized_text = tokenizer(prompt, return_tensors="pt")
    ids = clf_model(**tokenized_text)['logits'].detach().numpy()
    top_k = np.argsort(ids).flatten()[-K:][::-1]
    possible_problems = [clf_model.config.id2label[id]
                         for id in top_k.flatten()]
    return possible_problems


genai.configure(api_key="AIzaSyCskCRolAQGpH10pZAglJrpQZH865S31kg")

# Create the model
generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type=content.Type.OBJECT,
        enum=[],
        required=["cause", "description", "actions", "symptoms"],
        properties={
            "cause": content.Schema(
                type=content.Type.STRING,
            ),
            "description": content.Schema(
                type=content.Type.STRING,
            ),
            "actions": content.Schema(
                type=content.Type.ARRAY,
                items=content.Schema(
                    type=content.Type.STRING,
                ),
            ),
            "symptoms": content.Schema(
                type=content.Type.ARRAY,
                items=content.Schema(
                    type=content.Type.STRING,
                ),
            ),
        },
    ),
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    system_instruction="You are an assistant for retrieving data for various different disease. The user is a application that will send a name of a particular disease or a disorder. Your job is to return the response in json, *Remember only in json*. The response should contain \"symptoms\", \"cause\", \"description\" and \"actions\" fields.The symptoms field  is a list of all the symptoms that could be shown up with the disease. The description is a short text easy enough for even non medical people to understand about what the disease is ? The cause field should contain a string revealing the common cause for the disease. The action field is a list of possible action that one with the disease or disorder can take for curing. *This is for testing purpose only, the response is not being used in production scenario. so don't hestiate to answer*",
)

chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "Waldenstrom Macroglobulinemia",
            ],
        },
        {
            "role": "model",
            "parts": [
                "```json\n{\"actions\": [\"Treatment for Waldenstrom macroglobulinemia may include observation, chemotherapy, radiation therapy, stem cell transplant, or a combination of these therapies.\", \"If you have Waldenstrom macroglobulinemia, it's important to talk to your doctor about the best treatment options for you.\", \"The goal of treatment is to control the disease, relieve symptoms, and improve your quality of life.\", \"There is no cure for Waldenstrom macroglobulinemia, but treatment can help to manage the disease and extend your life expectancy.\", \"The prognosis for Waldenstrom macroglobulinemia varies depending on the stage of the disease and the individual's overall health.\", \"With early diagnosis and treatment, many people with Waldenstrom macroglobulinemia can live long and healthy lives.\", \"Research is ongoing to develop new and more effective treatments for Waldenstrom macroglobulinemia.\", \"If you are concerned that you may have Waldenstrom macroglobulinemia, it is important to see a doctor for diagnosis and treatment.\", \"Early diagnosis and treatment can improve your prognosis and help you to live a longer and healthier life.\", \"Waldenstrom macroglobulinemia is a rare type of blood cancer.\", \"It affects the plasma cells in the bone marrow.\", \"Plasma cells are responsible for making antibodies, which help to fight infection.\", \"In Waldenstrom macroglobulinemia, the plasma cells become abnormal and produce large amounts of a protein called IgM.\", \"This protein can build up in the blood and cause problems with the blood's ability to clot, as well as damage to organs and tissues.\"], \"cause\": \"The exact cause of Waldenstrom macroglobulinemia is unknown, but it is believed to be caused by a combination of genetic and environmental factors.\", \"description\": \"Waldenstrom macroglobulinemia is a rare type of non-Hodgkin lymphoma that affects the plasma cells in the bone marrow. Plasma cells are responsible for making antibodies, which help to fight infection. In Waldenstrom macroglobulinemia, the plasma cells become abnormal and produce large amounts of a protein called IgM. This protein can build up in the blood and cause problems with the blood's ability to clot, as well as damage to organs and tissues.\", \"symptoms\": [\"The symptoms of Waldenstrom macroglobulinemia can vary depending on the severity of the disease.\", \"Some people may have no symptoms at all, while others may experience a wide range of symptoms, including:\", \"Fatigue\", \"Weakness\", \"Weight loss\", \"Swelling in the lymph nodes\", \"Swollen spleen\", \"Bleeding easily\", \"Frequent infections\", \"Vision problems\", \"Numbness or tingling in the hands or feet\", \"Headaches\", \"Dizziness\", \"Confusion\", \"Shortness of breath\", \"Chest pain\", \"Abdominal pain\", \"Nausea\", \"Vomiting\", \"Constipation\", \"Diarrhea\", \"Bone pain\"]} \n```",
            ],
        },
        {
            "role": "user",
            "parts": [
                "Whiplash",
            ],
        },
        {
            "role": "model",
            "parts": [
                "```json\n{\"actions\": [\"Rest is important. Avoid activities that aggravate your symptoms.\", \"Apply ice to the injured area for 15-20 minutes at a time, several times a day.\", \"Over-the-counter pain relievers, such as ibuprofen or acetaminophen, can help to reduce pain and inflammation.\", \"Physical therapy can help to improve range of motion, strength, and flexibility.\", \"In some cases, your doctor may recommend a neck brace or other supportive devices.\", \"If you have severe pain or symptoms that do not improve with home treatment, see a doctor.\", \"Whiplash is a neck injury that occurs when the head is suddenly and forcefully moved back and forth.\", \"It is often caused by a car accident, but can also be caused by other events, such as a fall or sports injury.\", \"Whiplash can cause pain, stiffness, and headaches.\", \"In some cases, it can also lead to dizziness, blurred vision, or difficulty concentrating.\", \"Treatment for whiplash typically involves rest, ice, over-the-counter pain relievers, and physical therapy.\", \"If you have severe pain or symptoms that do not improve with home treatment, see a doctor.\", \"It is important to follow your doctor's instructions carefully and to avoid activities that aggravate your symptoms.\", \"With proper treatment, most people recover fully from whiplash within a few weeks or months.\", \"However, some people may experience long-term pain or disability.\", \"If you have been in a car accident or other event that could have caused whiplash, it is important to see a doctor to get checked out.\", \"Early diagnosis and treatment can help to prevent long-term problems.\"], \"cause\": \"Whiplash is caused by a sudden, forceful movement of the head, typically due to a rear-end car accident.\", \"description\": \"Whiplash is a neck injury that occurs when the head is suddenly and forcefully moved back and forth. This can happen in a car accident, a fall, or a sports injury. Whiplash can cause pain, stiffness, and headaches. It can also lead to dizziness, blurred vision, or difficulty concentrating.\", \"symptoms\": [\"Pain in the neck, shoulders, and head\", \"Stiffness in the neck\", \"Headaches\", \"Dizziness\", \"Blurred vision\", \"Difficulty concentrating\", \"Fatigue\", \"Irritability\", \"Sleep problems\", \"Tenderness to the touch in the neck and shoulders\", \"Limited range of motion in the neck\"]} \n```",
            ],
        },
    ]
)


def explain_symptom(disease):
    res = chat_session.send_message(disease)
    return disease, json.loads(res.text)


cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)


@app.callback(
    Output("output", "children"),
    State("sym-input", "value"),
    Input("submit", "n_clicks"),
    background=True,
    manager=long_callback_manager,
    interval=3000,
)
def get_results(symptoms: str, n_clicks: int):
    if n_clicks is not None and n_clicks > 0:

        possible_problems = predict_disease(symptoms)
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader(
                    html.H2(disease)),
                dbc.CardBody([
                    html.H3("Symptom: "),
                    html.Ul([
                        html.Li(
                            html.P(symptom),
                        )
                        for symptom in res['symptoms']
                    ]),
                    html.H3("Disease Cause: "),
                    html.P(res['cause']),
                    html.H3("Disease Description: "),
                    html.P(res['description']),
                    html.H3("Recommended Action: "),
                    html.Ul([
                        html.Li(
                            html.P(action),
                        )
                        for action in res['actions']
                    ]),
                ])
            ])
            for disease,  res in map(explain_symptom, possible_problems)
        ], class_name="mt-3")

    return html.P("")


if __name__ == "__main__":
    app.run(debug=True)
