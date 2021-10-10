import streamlit as st
from detect import main
import os
import pandas as pd
import base64
from PIL import Image


def app():
    background()
    thresh = sidebar()
    st.markdown("<h1 style='text-align: center; color:#99ffff;'>Apollo's Eye</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color:#99ffff;'>Realtime Crowd monitoring System</h3>",
                unsafe_allow_html=True)
    col1, col2, col3, col4 = st.beta_columns(5)

    with col1:
        pass
    with col2:
        center_button1 = st.button('Camera')
    with col4:
        pass
    with col3:
        center_button2 = st.button('Upload Video')

    if center_button1:
        if st.sidebar.button('SUBMIT'):
            main(conf_thres=thresh)
    if center_button2:
        if uploaded_file is not None:
            st.markdown("<h5 style='text-align: center; color:#99ffff;'>Data Loaded...</h5>",
                        unsafe_allow_html=True)
            with open(os.path.join(os.getcwd(), uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
                if st.sidebar.button('SUBMIT'):
                    main(source=os.path.join(os.getcwd(), uploaded_file.name), conf_thres=thresh)
                    os.remove(os.path.join(os.getcwd(), uploaded_file.name))
'''    if source_dict[source]:
        uploaded_file = st.file_uploader('Upload the required video file', type=['mp4', 'mkv'])
        if st.sidebar.button('SUBMIT'):
            if source_dict[source]:
                if uploaded_file is not None:
                    st.markdown("<h5 style='text-align: center; color:#99ffff;'>Data Loaded...</h5>",
                                unsafe_allow_html=True)
                    with open(os.path.join(os.getcwd(), uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    main(source = os.path.join(os.getcwd(), uploaded_file.name), conf_thres = thresh)
                    os.remove(os.path.join(os.getcwd(), uploaded_file.name))
            else:
                st.markdown("<h2 style='text-align: center; color:#99ffff;'>Upload the video to detect.</h2>",
                            unsafe_allow_html=True)

        else:
            main(source="0", conf_thres=thresh)
'''

def sidebar():
    side_bg = os.path.join(os.getcwd(), 'static/logo.jpg')
    side_bg_ext = "jpg"
    # st.markdown(
    #     f"""
    #         <style>
    #        .sidebar .sidebar-content {{
    #             background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    #         }}
    #          background-position: center;
    #       background-repeat: no-repeat;
    #     background-size: cover;
    #     }}
    #         </style>
    #         """,
    #     unsafe_allow_html=True
    # )
    st.markdown(
        """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    sidebar_logo = Image.open(side_bg)
    st.sidebar.image(sidebar_logo)
    st.sidebar.markdown("""# Customize View""")
    thresh = user_inputs()
    return thresh


def background():
    main_bg = os.path.join(os.getcwd(),'static/bg.jpg')
    main_bg_ext = "jpg"
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
         background-position: center;
      background-repeat: no-repeat;
    background-size: cover;
    }}
        </style>
        """,
        unsafe_allow_html=True
    )
    banner()


def banner():
    image = Image.open('static/logo.jpg')
    st.image(image)

def user_inputs():
    thresh = st.sidebar.slider('Threshold', 0.0, 1.0)
    #source = st.sidebar.selectbox("Media Source", ("Camera", "Uploaded Video"))
    return (thresh) #source)


if __name__ == '__main__':
    app()