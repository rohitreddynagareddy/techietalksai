import streamlit as st
import random
import math
import time
import graphviz

# --- Setup ---

st.set_page_config(page_title="Learn Perceptron", page_icon=":brain:", layout="wide")

st.title("Perceptron Playground :brain:")
st.subheader("Understand Neural Networks Visually")

# --- Global Variables ---

problems = {
    "AND Gate": {
        "inputs": ["Input 1", "Input 2"],
        "dataset": [
            ([0, 0], 0),
            ([0, 1], 0),
            ([1, 0], 0),
            ([1, 1], 1),
        ]
    },
    "OR Gate": {
        "inputs": ["Input 1", "Input 2"],
        "dataset": [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 1),
        ]
    },
    "Exercise Decision": {
        "inputs": ["Mother", "Father", "Sister", "Brother", "Teacher"],
        "dataset": [
            ([0, 0, 0, 0, 0], 0),
            ([1, 0, 0, 0, 0], 0),
            ([0, 1, 0, 0, 0], 1),
            ([0, 0, 1, 0, 0], 0),
            ([0, 0, 0, 1, 0], 1),
            ([0, 0, 0, 0, 1], 1),
            ([1, 1, 1, 1, 1], 1),
            ([1, 0, 1, 0, 0], 1),
        ]
    },
    "Rain Prediction": {
        "inputs": ["Cloudy", "Monsoon Season", "High Humidity", "Windy", "Cool Temperature"],
        "dataset": [
        ([1, 1, 1, 0, 1], 1),  # Cloudy, Monsoon, Humid, Not windy, Cool => Rain
        ([1, 0, 1, 1, 0], 1),  # Cloudy, No Monsoon, Humid, Windy, Warm => Rain
        ([0, 1, 0, 0, 1], 1),  # Not Cloudy, Monsoon, No Humid, No Wind, Cool => Rain
        ([1, 0, 0, 1, 1], 0),  # Cloudy, No Monsoon, Not humid, Windy, Cool => No Rain
        ([0, 0, 1, 0, 0], 0),  # Not Cloudy, No Monsoon, Humid, No Wind, Warm => No Rain
        ([0, 0, 0, 0, 1], 0),  # Clear, No Monsoon, No Humid, No Wind, Cool => No Rain
        ([1, 1, 1, 1, 1], 1),  # Everything active => Rain
        ([0, 1, 0, 1, 0], 1),  # Monsoon and Windy => Rain
        ]
    }
}

if "problem" not in st.session_state:
    st.session_state.problem = "AND Gate"
if "weights" not in st.session_state:
    st.session_state.weights = [random.uniform(-1, 1) for _ in range(2)]
if "bias" not in st.session_state:
    st.session_state.bias = random.uniform(-1, 1)
if "learning_rate" not in st.session_state:
    st.session_state.learning_rate = 0.1
if "message" not in st.session_state:
    st.session_state.message = ""

if "history" not in st.session_state:
    st.session_state.history = []


# --- Functions ---

# def activation(z):
#     return 1 if z >= 0 else 0

def activation(z):
    if "activation_choice" not in st.session_state:
        st.session_state.activation_choice = "Step"

    activation_choice = st.session_state.activation_choice

    if activation_choice == "Step":
        return 1 if z >= 0 else 0
    
    elif activation_choice == "Stochastic":
        probability = 1 / (1 + math.exp(-z))  # Sigmoid
        return 1 if random.random() < probability else 0

    else:
        return 1 if z >= 0 else 0
def reset_problem(name):
    st.session_state.problem = name
    inputs = problems[name]["inputs"]
    st.session_state.weights = [random.uniform(-1, 1) for _ in range(len(inputs))]
    st.session_state.bias = random.uniform(-1, 1)
    st.session_state.message = "New Problem Selected! Weights and Bias reset."

def train_step():
    dataset = problems[st.session_state.problem]["dataset"]

    total_error = 0
    for inputs, target in dataset:
        weighted_sum = sum(x * w for x, w in zip(inputs, st.session_state.weights)) + st.session_state.bias
        output = activation(weighted_sum)
        error = target - output

        # Update weights and bias
        for i in range(len(st.session_state.weights)):
            st.session_state.weights[i] += st.session_state.learning_rate * error * inputs[i]
        st.session_state.bias += st.session_state.learning_rate * error

        total_error += abs(error)

    # Create log for this epoch
    epoch_output = f"Epoch {len(st.session_state.history) + 1}: "
    epoch_output += ", ".join([f"w{i+1}={w:.2f}" for i, w in enumerate(st.session_state.weights)])
    epoch_output += f", bias={st.session_state.bias:.2f}, Mistakes made = {total_error}"

    st.session_state.history.append(epoch_output)
    st.session_state.message = "One Epoch Trained! (See Training Log Below)"
    time.sleep(2)

def draw_perceptron_diagram():
    inputs = problems[st.session_state.problem]["inputs"]
    weights = st.session_state.weights
    bias = st.session_state.bias

    import graphviz

    g = graphviz.Digraph()

    # Set Graph (global) styles
    g.attr(bgcolor="#4f777f")  # Dark background (Streamlit dark theme)
    g.attr('node', style='filled', fillcolor="#262730", fontcolor="white", fontname="Arial", fontsize="16")
    g.attr('edge', color="white", fontcolor="white", fontsize="14", penwidth="3")

    # Inputs with Weights
    for i, name in enumerate(inputs):
        g.node(f"input{i}", f"{name}\n(w={st.session_state.weights[i]:.2f})", shape='ellipse')
        g.edge(f"input{i}", "neuron", label=f"{st.session_state.weights[i]:.2f}")

    # Bias
    g.node("bias", f"Bias\n({st.session_state.bias:.2f})", shape='rectangle')
    g.edge("bias", "neuron", label=f"{st.session_state.bias:.2f}")

    # Neuron
    activation = st.session_state.activation_choice
    if activation == "Step":
        act = "Step(z≥0)"
    else:
        act = "Stochastic (random sigmoid probability)"

    g.node("neuron", f"Neuron\nΣ(inputs*w) + bias\nActivation: {act}", shape="circle")

    # Output
    output_symbol = "✅" if output == 1 else "❌"
    g.node("output", f"Output\n{output_symbol}", shape="doublecircle")
    g.edge("neuron", "output", label="Activation")

    # Display
    st.graphviz_chart(g)


# def train_fully():
#     dataset = problems[st.session_state.problem]["dataset"]
    
#     for _ in range(50):
#         total_error = 0
#         for inputs, target in dataset:
#             weighted_sum = sum(x * w for x, w in zip(inputs, st.session_state.weights)) + st.session_state.bias
#             output = activation(weighted_sum)
#             error = target - output

#             for i in range(len(st.session_state.weights)):
#                 st.session_state.weights[i] += st.session_state.learning_rate * error * inputs[i]
#             st.session_state.bias += st.session_state.learning_rate * error

#             total_error += abs(error)

#         # Create log for this epoch
#         epoch_output = f"Epoch {len(st.session_state.history) + 1}: "
#         epoch_output += ", ".join([f"w{i+1}={w:.2f}" for i, w in enumerate(st.session_state.weights)])
#         epoch_output += f", bias={st.session_state.bias:.2f}, error={total_error}"

#         st.session_state.history.append(epoch_output)

#         if total_error == 0:
#             break

#     st.session_state.message = f"Training Completed! (See Full Log Below)"
#     time.sleep(2)
def train_fully():
    dataset = problems[st.session_state.problem]["dataset"]

    placeholder = st.empty()

    for epoch in range(50):
        total_error = 0
        for inputs, target in dataset:
            weighted_sum = sum(x * w for x, w in zip(inputs, st.session_state.weights)) + st.session_state.bias
            output = activation(weighted_sum)
            error = target - output

            for i in range(len(st.session_state.weights)):
                st.session_state.weights[i] += st.session_state.learning_rate * error * inputs[i]
            st.session_state.bias += st.session_state.learning_rate * error

            total_error += abs(error)

        # Create log for this epoch
        epoch_output = f"Epoch {len(st.session_state.history) + 1}: "
        epoch_output += ", ".join([f"w{i+1}={w:.2f}" for i, w in enumerate(st.session_state.weights)])
        epoch_output += f", bias={st.session_state.bias:.2f}, Mistakes made = {total_error}"

        st.session_state.history.append(epoch_output)

        # --- Update the UI dynamically
        with placeholder.container():
            st.subheader(f"Training Progress (Epoch {epoch+1})")
            for line in st.session_state.history:
                st.write(line)

        time.sleep(2)  # Small delay to see change

        if total_error == 0:
            break

    st.session_state.message = f"Training Completed! (See Full Log Below)"


# --- UI Layout ---

col1, col2 = st.columns(2)

with col1:
    st.header("Settings")

    selected_problem = st.selectbox("Choose a Problem", list(problems.keys()), index=list(problems.keys()).index(st.session_state.problem))
    if selected_problem != st.session_state.problem:
        reset_problem(selected_problem)
        st.session_state.history.clear()  # Clear log when new problem selected
        st.rerun()


    st.write("---")
    st.subheader("Training Dataset (Truth Table)")
    # Get the selected problem's dataset
    dataset = problems[st.session_state.problem]["dataset"]
    inputs_names = problems[st.session_state.problem]["inputs"]

    # Prepare data for display
    table_data = []
    for inputs, target in dataset:
        row = {inputs_names[i]: inputs[i] for i in range(len(inputs))}
        row["Target Output"] = target
        table_data.append(row)

    # Display as a table
    st.table(table_data)


    st.write("---")
    if st.button("Train Step"):
        train_step()

    if st.button("Train Fully"):
        train_fully()


    st.write("---")
    st.success(st.session_state.message)

    # Display all training logs
    if st.session_state.history:
        st.subheader("Training Log")
        for line in st.session_state.history:
            st.write(line)
import graphviz  # Make sure this is imported at top if not already

with col2:
    st.header("Inputs and Output")

    inputs = problems[st.session_state.problem]["inputs"]
    user_inputs = []

    input_col, spacer_col, output_col = st.columns([1.5, 0.2, 1])

    with input_col:
        st.subheader("Inputs")
        for i, name in enumerate(inputs):
            label = f"{name} (w={st.session_state.weights[i]:.2f})"
            val = st.checkbox(label, value=False, key=f"input_{i}")
            user_inputs.append(1 if val else 0)

    weighted_sum = sum(x * w for x, w in zip(user_inputs, st.session_state.weights)) + st.session_state.bias
    output = activation(weighted_sum)

    with output_col:
        st.subheader("Output")
        if output == 1:
            st.markdown("<h1 style='text-align: center; color: green;'>✅</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; color: red;'>❌</h1>", unsafe_allow_html=True)

    st.write("---")

    st.subheader("Activation Function Settings")
    activation_choice = st.selectbox(
        "Choose Activation Function",
        ["Step", "Stochastic"]
    )
    
    if "activation_choice" not in st.session_state:
        st.session_state.activation_choice = "Step"
    st.session_state.activation_choice = activation_choice

    st.write("---")
    st.subheader("Perceptron Calculation")
    for i, name in enumerate(inputs):
        st.write(f"{name}: {user_inputs[i]} * {st.session_state.weights[i]:.2f} = {user_inputs[i]*st.session_state.weights[i]:.2f}")

    

    weighted_sum_only = sum(x * w for x, w in zip(user_inputs, st.session_state.weights))
    weighted_sum_with_bias = weighted_sum_only + st.session_state.bias

    st.write(f"Weighted Sum (without bias): {weighted_sum_only:.2f}")
    st.write(f"Bias: {st.session_state.bias:.2f}")
    st.write(f"Weighted Sum + Bias: {weighted_sum_with_bias:.2f}")

    st.write(f"Activation Function: **{st.session_state.activation_choice} Function**")
    st.markdown("```If (Weighted Sum + Bias) >= 0, Output = 1; else 0```")

    st.write(f"Activation Output: {'1 (Yes)' if output == 1 else '0 (No)'}")

    st.write("---")
    st.subheader("Weights and Biases")
    for i, w in enumerate(st.session_state.weights):
        st.write(f"w{i+1}: {w:.2f}")
    st.write(f"bias: {st.session_state.bias:.2f}")




st.write("---")
st.subheader("Perceptron Diagram")
draw_perceptron_diagram()

    # g = graphviz.Digraph()

    # # Inputs with Weights
    # for i, name in enumerate(inputs):
    #     g.node(f"input{i}", f"{name}\n(w={st.session_state.weights[i]:.2f})")
    #     g.edge(f"input{i}", "neuron", label=f"{st.session_state.weights[i]:.2f}")

    # # Bias
    # g.node("bias", f"Bias\n({st.session_state.bias:.2f})", shape='rectangle')
    # g.edge("bias", "neuron", label=f"{st.session_state.bias:.2f}")

    # # Neuron with activation function mentioned
    # g.node("neuron", "Neuron\nΣ(inputs*w) + bias\nActivation: Step(z≥0)", shape="circle")

    # # Output
    # output_symbol = "✅" if output == 1 else "❌"
    # g.node("output", f"Output\n{output_symbol}", shape="doublecircle")
    # g.edge("neuron", "output", label="Activation")

    # st.graphviz_chart(g)


