from transformers import pipeline
import streamlit as st
import json
from fuzzywuzzy import process

@st.cache_data
def load_herbal_data():
    with open("arjun.json", "r") as file:
        return json.load(file)

herbal_data = load_herbal_data()

def search_herbal_data(query):
    # Extract only the top match based on confidence score
    all_plants = [(plant['name'], plant) for category, plants in herbal_data.items() for plant in plants]
    matches = process.extract(query, [x[0] for x in all_plants], limit=1)  # Return only one result
    
    if matches:
        best_match = matches[0]
        plant = next(plant for name, plant in all_plants if name == best_match[0])
        plant['confidence'] = best_match[1]  # Add confidence score
        return plant
    return None

qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

def get_qa_response(query):
    context = "\n".join(
        [
            f"{plant['name']} ({plant['scientific_name']}): {plant['medical_use']}"
            for category, plants in herbal_data.items()
            for plant in plants
        ][:1000]  # Limit context size
    )
    try:
        result = qa_model(question=query, context=context)
        return result.get("answer", "No answer found.")
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333;
    }
    .stApp {
        background: linear-gradient(to bottom, #8bc34a, #e8f5e9);
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #fff;
        margin-top: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #fff;
        margin-bottom: 20px;
    }
    .result {
        color: #333;  
        background-color: #ffffffcc;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .stButton > button {
        background-color: #4caf50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        margin: 0 10px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🌿 HERBSPHERE AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">वनस्पति नां रहस्यं वि मो चयन्तु</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explore the world of herbal plants and their uses!</div>', unsafe_allow_html=True)

# Input field
query = st.text_input("Type your query below:")

# Buttons for predefined queries
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("What is the use of Neem?"):
    query = "What is the use of Neem?"
if st.button("Benefits of Turmeric?"):
    query = "Turmeric?"
if st.button("how to plant and grow turmeric plants?"):
    query = "how to plant and grow turmeric plants?"
st.markdown('</div>', unsafe_allow_html=True)

# Processing the query
if query:
    plant = search_herbal_data(query)
    
    if plant:
        st.write("### HERBSPRHERE bot:")
        st.markdown(
            f"""
            <div class="result">
            <b>🌱 Name:</b> {plant['name']}<br>
            <b>Scientific Name:</b> {plant['scientific_name']}<br>
            <b>Medical Uses:</b> {plant['medical_use']}<br>
            <b>Availability:</b> {', '.join(plant['availability'])}<br>
            <b>Native Region:</b> {plant['Habitat'].get('Native Region', 'Unknown')}<br>
            <b>Growth Regions:</b> {plant['Habitat'].get('Growth Regions', 'Unknown')}<br>
            <b>Climate:</b> {plant['Method of Cultivation'].get('Climate', 'Unknown')}<br>
            <b>Soil:</b> {plant['Method of Cultivation'].get('Soil', 'Unknown')}<br>
            <b>Propagation:</b> {plant['Method of Cultivation'].get('Propagation', 'Unknown')}<br>
            <b>Planting:</b> {plant['Method of Cultivation'].get('Planting', 'Unknown')}<br>
            <b>Care:</b> {plant['Method of Cultivation'].get('Care', 'Unknown')}<br>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "image_url" in plant:  # Display image if available
            st.image(plant["image_url"], caption=f"{plant['name']} - Main Image", use_container_width=True)
        if "new_url" in plant:  # Display additional image
            st.image(plant["new_url"], caption=f"{plant['name']} - Additional Image", use_container_width=True)
    else:
        st.write("### AI-Powered Response:")
        with st.spinner("Thinking..."):
            response = get_qa_response(query)
        st.write(response)
else:
    st.write("Type a query or click a button to get started!")
