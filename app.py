import streamlit as st

# Set page configuration
st.set_page_config(page_title="🏪 Retail Analytics", layout="wide")

# Inject custom CSS for background image only on the app page (not sidebar)
st.markdown("""
    <style>
        body {
            background-image: url('https://your-image-url.com/background.jpg');  /* Replace with your image URL */
            background-size: cover;  /* This will make the image cover the entire page */
            background-position: center;
            background-repeat: no-repeat;
        }
        
        /* Styling for the team members list to position it at the top right corner */
        .team-members {
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: rgba(255, 255, 255, 0.7);  /* Slightly transparent white background */
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }

        /* Customizing the title and the rest of the text */
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #F63366;  /* Customize this color as per your brand */
        }
    </style>
""", unsafe_allow_html=True)

# Add logo image at the top (adjust to your logo size)
st.image("https://www.upsite.com/wp-content/uploads/2020/01/how-data-centers-and-the-retail-industry-affect-each-other.jpeg", width=200)

# Title of the app
st.markdown('<p class="title">🏪 Welcome to Retail Analytics Dashboard</p>', unsafe_allow_html=True)

# Instructions for navigation
st.markdown("""
Use the sidebar to navigate between:
- **Data Loader**
- **Search Household**
- **CLV Dashboard**
""")

# Add Team Members section at the top right corner using custom CSS class
st.markdown("""
    <div class="team-members">
        <strong>Team Members:</strong>
        <ul><titile>Team Members</titile>
            <li>Amrutha Reddy Gurugari</li>
            <li>Ramakrishna Gampa</li>
            <li>Saketh Patel Koneru</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# You can add more content and components below this as per your app's requirements
