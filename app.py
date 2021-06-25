import streamlit as st
# ML Pkg
# from sklearn.externals import joblib
import joblib
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
    unsafe_allow_html=True,
)
#######
interactive = st.beta_container()


# load Vectorizer For News Prediction
vectorizer = open("tfidf.pkl", "rb")
cv = joblib.load(vectorizer)

# load Model For News Prediction
svm_model = open("model.pkl", "rb")
clf = joblib.load(svm_model)


# Prediction
def predict_news(data):
    vect = cv.transform(data).toarray()
    result = clf.predict(vect)
    return result


def main():

    html_temp = """
	<div style="background-color:tomato;padding:10px">
	<h2 style="color:white;text-align:center;">Covid-19 Fake News Detection </h2>
	</div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    name = st.text_input("Enter News", "Type Here")
    if st.button("Predict"):
        result = predict_news([name])
        if (result >= 0.5):

            st.success('Genuine')

        else:

            st.error('Fake')

    if st.button("Dataset"):

        with interactive:
            news_data = pd.read_csv('news.csv')
            fig = go.Figure(data=go.Table(columnwidth=[5, 1],   header=dict(values=list(news_data[['News', 'Outcome']].columns), fill_color='OliveDrab', align='center'), cells=dict(
                values=[news_data.News, news_data.Outcome], fill_color='Black', align='center')))

        fig.update_layout(margin=dict(l=5, r=5, b=10, t=10))

        st.write(fig)

    if st.button("About"):

        st.text("Rishika Ahuja")
        st.text("Ann Elizabeth Zachariah")
        st.text("Adil Ansari ")


if __name__ == '__main__':
    main()
