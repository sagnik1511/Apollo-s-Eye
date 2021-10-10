import streamlit as st
from detect import main
import os
import base64
from PIL import Image

source_dict = {
    "Camera": False,
    "Uploaded Video": True
}


def app():
    background()
    thresh, source = sidebar()
    st.markdown("<h3 style='text-align: center; color:#99ffff;'>Realtime Crowd Monitoring System</h3>",
                unsafe_allow_html=True)
    if source_dict[source]:
        uploaded_file = st.file_uploader('Upload the required video file', type=['mp4', 'mkv'])
    if st.sidebar.button('SUBMIT'):
        if source_dict[source]:
            if uploaded_file is not None:
                st.markdown("<h5 style='text-align: center; color:#99ffff;'>Data Loaded...</h5>",
                            unsafe_allow_html=True)
                with open(os.path.join(os.getcwd(), uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                main(source = os.path.join(os.getcwd(), uploaded_file.name), conf_thresh = thresh)
                os.remove(os.path.join(os.getcwd(), uploaded_file.name))
            else:
                st.markdown("<h2 style='text-align: center; color:#99ffff;'>Upload the video to detect.</h2>",
                        unsafe_allow_html=True)
        else:
            main(source="0", conf_thresh=thresh)


def sidebar():
    side_bg = os.path.join(os.getcwd(), 'static/logo.jpg')
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
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        image = Image.open('static/banner.png')
        st.image(image, width=500)

    with col3:
        st.write("")

def buttons():
    col1, col2, col3 = st.columns([13,6,10])

    with col1:
        st.write("")

    with col2:
        stop = st.button('STOP')

    with col3:
        st.write("")

    return stop

def user_inputs():
    thresh = st.sidebar.slider('Threshold', 0.0, 1.0, 0.35)
    source = st.sidebar.selectbox("Media Source", ("Camera","Uploaded Video"))
    return (thresh,source)

if __name__ == '__main__':
    app()
