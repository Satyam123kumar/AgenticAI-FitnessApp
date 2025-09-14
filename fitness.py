import os
from typing import TypedDict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
import streamlit as st

# --- 1. Set Up Environment & Page Configuration ---
# For deployment on Vercel, use Streamlit's secrets management.
# The GOOGLE_API_KEY must be set as an environment variable in your Vercel project settings.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except (KeyError, FileNotFoundError):
    # This block allows for local development without secrets.toml
    # but will show an error if the key isn't set in deployment.
    GOOGLE_API_KEY = "" # Set to empty string if not found


# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Health & Fitness Plan", page_icon="🏋️‍♂️", layout="wide")

# --- 2. Define the State for the Graph ---
# The state is a shared dictionary that each node in the graph can read from and write to.
class AgentState(TypedDict):
    """Defines the structure of the data that flows through the graph."""
    name: str
    age: int
    weight: float
    height: float
    activity_level: str
    dietary_preference: str
    fitness_goal: str
    # Each node's output is added to the state. `Optional` means they can be empty initially.
    meal_plan: Optional[str]
    fitness_plan: Optional[str]
    holistic_plan: Optional[str]


# --- 3. Define Tools and the LLM ---
# This tool can be used by any agent/node in the graph.
search_tool = DuckDuckGoSearchRun()
# Initialize the Large Language Model. We'll use the same one for all nodes.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")


# --- 4. Define the Agent Nodes ---
# Each node is a function that takes the current state and returns a dictionary
# with the fields to update in the state.

def dietary_planner_node(state: AgentState, config: RunnableConfig):
    """
    This node generates a personalized meal plan based on user details.
    It can use the DuckDuckGo tool to search for information if needed.
    """
    print("--- Executing Dietary Planner Node ---")
    
    # Define the specific prompt for this agent
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a specialized dietary planner. Your task is to create a personalized dietary plan based on the user's information. "
         "Instructions:\n"
         "- Generate a detailed diet plan including breakfast, lunch, dinner, and snacks.\n"
         "- Consider the user's dietary preferences (e.g., Keto, Vegetarian, Low Carb).\n"
         "- Ensure the plan includes proper hydration and electrolyte balance tips.\n"
         "- Provide a nutritional breakdown (macronutrients, vitamins).\n"
         "- Suggest meal preparation tips for easy implementation.\n"
         "- You have access to a web search tool if you need additional information for specific food items or nutritional data."),
        ("user",
         "Please create a meal plan for the following person:\n"
         f"- Age: {state['age']}\n"
         f"- Weight: {state['weight']}kg\n"
         f"- Height: {state['height']}cm\n"
         f"- Activity Level: {state['activity_level']}\n"
         f"- Dietary Preference: {state['dietary_preference']}\n"
         f"- Fitness Goal: {state['fitness_goal']}")
    ])
    
    # Create a runnable chain: Prompt -> LLM with Tool -> Output
    llm_with_tools = llm.bind_tools([search_tool])
    chain = prompt_template | llm_with_tools
    
    # Invoke the chain and get the meal plan
    meal_plan_result = chain.invoke({}, config=config)
    return {"meal_plan": meal_plan_result.content}


def fitness_trainer_node(state: AgentState, config: RunnableConfig):
    """
    This node generates a customized workout routine.
    It can also use the search tool for specific exercise information.
    """
    print("--- Executing Fitness Trainer Node ---")
    
    # Define the prompt for the fitness trainer
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert fitness trainer. Your goal is to generate a customized workout routine based on the user's fitness goals and details. "
         "Instructions:\n"
         "- Create a workout plan that includes warm-ups, main exercises (with sets/reps), and cool-downs.\n"
         "- Adjust the workout intensity based on the user's stated fitness level or infer it from their activity level.\n"
         "- Tailor the plan to their specific goal (e.g., weight loss, muscle gain, endurance).\n"
         "- Provide clear safety tips and advice for injury prevention.\n"
         "- Suggest methods for tracking progress to maintain motivation.\n"
         "- You can use the web search tool to find information about specific exercises or fitness concepts."),
        ("user",
         "Please generate a workout plan for the following person:\n"
         f"- Age: {state['age']}\n"
         f"- Weight: {state['weight']}kg\n"
         f"- Height: {state['height']}cm\n"
         f"- Activity Level: {state['activity_level']}\n"
         f"- Fitness Goal: {state['fitness_goal']}")
    ])
    
    # Create the runnable chain
    llm_with_tools = llm.bind_tools([search_tool])
    chain = prompt_template | llm_with_tools
    
    # Invoke the chain to get the fitness plan
    fitness_plan_result = chain.invoke({}, config=config)
    return {"fitness_plan": fitness_plan_result.content}


def team_lead_node(state: AgentState, config: RunnableConfig):
    """
    This final node acts as the team lead. It takes the generated meal and fitness plans,
    combines them into a holistic strategy, and adds motivational tips.
    """
    print("--- Executing Team Lead Node ---")
    
    # Define the prompt for the team lead/aggregator
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are the team lead for a health and wellness service. Your role is to combine the specialized diet and workout plans into a single, cohesive, and motivational message for the client. "
         "Instructions:\n"
         "- Start with a warm greeting to the user by their name.\n"
         "- Present the meal plan and workout plan clearly, using markdown tables and distinct sections for readability.\n"
         "- Ensure the plans are aligned and explain how the diet supports the workout goals.\n"
         "- Add a section with lifestyle tips for motivation, consistency, and overall well-being.\n"
         "- Provide guidance on how to track progress and when to consider adjusting the plans."),
        ("user",
         "Client Name: {name}\n\n"
         "User Information: {age} years old, {weight}kg, {height}cm, activity level: {activity_level}.\n"
         "Fitness Goal: {fitness_goal}\n\n"
         "Here is the generated Meal Plan:\n"
         "```markdown\n{meal_plan}\n```\n\n"
         "And here is the generated Workout Plan:\n"
         "```markdown\n{fitness_plan}\n```\n\n"
         "Please combine these into a holistic health strategy and present it to the client.")
    ])
    
    # Create the final chain (no tools needed for this node)
    chain = prompt_template | llm
    
    # Invoke the chain with the necessary data from the state
    holistic_plan_result = chain.invoke({
        "name": state["name"],
        "age": state["age"],
        "weight": state["weight"],
        "height": state["height"],
        "activity_level": state["activity_level"],
        "fitness_goal": state["fitness_goal"],
        "meal_plan": state["meal_plan"],
        "fitness_plan": state["fitness_plan"]
    }, config=config)
    
    return {"holistic_plan": holistic_plan_result.content}


# --- 5. Build the Graph ---
# This is where we define the workflow and how the nodes are connected.

# Create a new graph instance
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("dietary_planner", dietary_planner_node)
workflow.add_node("fitness_trainer", fitness_trainer_node)
workflow.add_node("team_lead", team_lead_node)

# Set the entry point of the graph. Execution starts here.
workflow.set_entry_point("dietary_planner")
workflow.add_edge("dietary_planner", "fitness_trainer")
workflow.add_edge("fitness_trainer", "team_lead")
workflow.add_edge("team_lead", END)

# Compile the graph into a runnable application
app = workflow.compile()


# --- 6. Streamlit User Interface ---

# Custom Styles for a Fitness and Health Theme
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #FF6347; /* Tomato Red */
        }
        .subtitle {
            text-align: center;
            font-size: 24px;
            color: #4CAF50; /* Green */
        }
        .goal-card {
            padding: 20px;
            margin: 10px;
            background-color: #FFF;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #FF6347;
            color: #333; /* Set a dark text color for readability */
        }
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown('<h1 class="title">🏋️‍♂️ AI Health & Fitness Plan Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Personalized fitness and nutrition plans to help you achieve your health goals!</p>', unsafe_allow_html=True)

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Your Health & Fitness Profile")

# User inputs for personal information and fitness goals
age = st.sidebar.number_input("Age (in years)", min_value=10, max_value=100, value=25)
weight = st.sidebar.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
height = st.sidebar.number_input("Height (in cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
activity_level = st.sidebar.selectbox("Activity Level", ["Low", "Moderate", "High"])
dietary_preference = st.sidebar.selectbox("Dietary Preference", ["Balanced", "Keto", "Vegetarian", "Low Carb", "Vegan"])
fitness_goal = st.sidebar.selectbox("Fitness Goal", ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility", "General Fitness"])


# --- Main Content Area ---
st.markdown("---")
name = st.text_input("What's your name?", "Alex")

# Button to generate the full health plan, now in the sidebar
if st.sidebar.button("🚀 Generate My Health Plan"):
    if not all([age, weight, height, name]):
        st.sidebar.warning("Please fill in all required fields.")
    elif not GOOGLE_API_KEY:
        st.error("Google API Key is not configured. Please set it in your Vercel environment variables.")
    else:
        with st.spinner("💥 Generating your personalized health & fitness plan... This may take a moment."):
            try:
                # The input for the graph must match the structure of AgentState
                inputs = {
                    "name": name,
                    "age": age,
                    "weight": weight,
                    "height": height,
                    "activity_level": activity_level,
                    "dietary_preference": dietary_preference,
                    "fitness_goal": fitness_goal,
                }
                
                # Invoke the graph with the user's input
                final_state = app.invoke(inputs)
                
                # Display the generated health plan in the main section
                st.subheader(f"Your Personalized Health & Fitness Plan, {name}!")
                st.markdown(final_state['holistic_plan'])

                st.balloons() # Fun animation on success!
                
                st.info("This is your customized health and fitness strategy, including meal and workout plans.")

                # Motivational Message
                st.markdown("""
                    <div class="goal-card">
                        <h4>🏆 Stay Focused, Stay Fit!</h4>
                        <p>Consistency is key! Keep pushing yourself, and you will see results. Your fitness journey starts now!</p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during plan generation: {e}")

