
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
# %matplotlib inline
import plotly.figure_factory as ff

import streamlit as st 
from streamlit_option_menu import option_menu

import sqlite3
import base64
from PIL import Image

import pickle as pk
import joblib as jb
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# text to speech library

# import pyttsx3
# engine=pyttsx3.init()
# engine.say("Heart Disease Prediction System")
# engine.say("By Amare Abayneh")
# engine.say("ID: PRAMIT/010/14")
# engine.runAndWait()
# if engine._inLoop:
#     engine.endLoop()

class heart_prediction:
    def __init__(self) -> None:

        @st.cache(allow_output_mutation=True)
        def get_base64_of_bin_file(bin_file):
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()

        def set_png_as_page_bg(png_file):
            bin_str = get_base64_of_bin_file(png_file)
            page_bg_img = '''
            <style>
            .stApp {
                background-image: url("data:image/png;base64,%s");
                background-size: contain;
                background-repeat: no-repeat;
                background-attachment: scroll; 
                background-size: cover;
            }
            </style>
            ''' % bin_str      
            st.markdown(page_bg_img, unsafe_allow_html=True)
            return

        set_png_as_page_bg('assets/image/2-2.png')

        # loading dataset
        self.df=pd.read_csv("heart dataset after prep/heart_dataset.csv")
        self.st=StandardScaler()
        # taking care of missing value
        data=self.df.drop_duplicates(subset=None, keep='first', inplace=False,ignore_index=True)

        # spilting dataset
        self.x=data.drop("target",axis=1)
        self.y=data["target"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = 0.3, random_state = 0)
        self.x_train=self.st.fit_transform(self.x_train)
        self.x_test=self.st.transform(self.x_test)
        
        self.knn=KNeighborsClassifier(n_neighbors=8)
        self.knn.fit(self.x_train,self.y_train)
        # p=self.knn.predict(self.x_train,self.x_test)
        # print(p)    
        jb.dump(self.knn,'heart.joblib')
    
    def main(self):
        selected=option_menu(
                menu_title=None,
                options=["Home","Prediction","about us","contact"],
                icons=["house","activity","file-earmark-person","person-rolodex"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
                
                styles={
                "container": {"padding": "0!important", "background-color": "#000033"},
                "icon": {"color": "orange", "font-size": "20px"}, 
                "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "gray",
                            "color":"white"  },
                "nav-link-selected": {"background-color": "green","color":"white"},
            }
                )


        if selected=="Home":
            st.header("HEART DISEASE PREDICTION SYSTEM")
            image = Image.open('assets/image/2.jfif')
            st.image(image, caption='Sunrise by the mountains')
            st.write("""
                    Heart disease describes a range of conditions that affect your heart. 
                    Heart disease term includes a number of diseases such as blood vessel diseases, 
                    such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you're born with (congenital heart defects), among others. 
                    The term heart disease is sometimes used interchangeably with the term cardiovascular disease. 
                    Cardiovascular disease (CVD) generally refers to conditions that involve narrowed or blocked blood vessels that can lead to a heart attack (Myocardial infarctions), chest pain (angina) or stroke. Other heart conditions, 
                    such as those that affect your heart's muscle, valves or rhythm, also are considered forms of heart disease. 17.9 million People die each year from CVDs, an estimated 31% of all deaths worldwide.

                """)
            
        elif selected=="Prediction":
            

            choice=["Prediction","EDA","Dataset Description"]
            task=st.sidebar.selectbox("Select Activity",options=choice)
            if task=="Prediction":
                # st.header("Prediction System")

                checkbtn=st.checkbox("Display sample dataset")
                if checkbtn:
                    df=pd.read_csv("heart dataset after prep/heart_dataset.csv")
                    st.write(df.sample(5,random_state=2))


                
                sex_options=["Male","Female"]
                cp_options=['Typical angina','Atypical angina','Non-angina pain','Asymptomatic']
                fbs_options=["Yes","No"]
                rest_ecg_options=['Nothing to note','ST-T Wave abnormality','left ventricular hypertropy']
                exang_options=["Yes","No"]
                slope_options=['Upsloping','Flatsloping','Downsloping']
                ca_options=['0','1','2','3']
                thal_options=['fixed defect','normal','reversable defect']

                
                st.subheader("New Input Prediction")
                col1, col2, col3=st.columns(3)

                age=col1.number_input("Age", min_value=29, max_value=77, value=29, step=1)
                gender=col1.selectbox("Gender [1:Male,0:Female]",options=sex_options)
                cp=col1.selectbox("Chest Pain Type [0:Typical-3:Asymptomatic]",options=cp_options)
                bp=col1.number_input("Rest Blood Pressure", min_value=94, max_value=200, step=1)
                chol=col1.number_input("Cholestrol in mg/dl", min_value=126, max_value=564, value=128, step=1)
                
                fbs=col2.selectbox("Fasting Blood Suger >120 mg/dl [0:No,1:Yes]",options=fbs_options)
                rest_ecg=col2.selectbox("Resting Electrocardiographic [0:Nothing-2:left ventricular]",options=rest_ecg_options)
                thalach=col2.number_input("Maximum Heart Rate Achieved", min_value=71, max_value=202, value=71, step=1)
                exang=col2.selectbox("Exercise Induced Angina [0:No,1:yes]",options=exang_options)
                
                old_peak=col3.number_input("Old Peak")
                slope=col3.selectbox("Slope [0:Upsloping,1:Flatsloping,2:Downsloping]",options=slope_options)
                ca=col3.selectbox("Number of major vessel",options=ca_options)
                thal=col3.selectbox("Thalasemia [1:fixed defect,2:normal ,3:reversable defect]",options=thal_options)
                

                btn_col1,btn_col2=st.columns(2)
                if btn_col1.button("Predict"):
                    if gender=="Male":
                        gender=1
                    else:
                        gender=0
                    if cp =='Typical angina':
                        cp=0
                    elif cp =='Atypical angina':
                        cp=1
                    elif cp =='Non-anginal pain':
                        cp=2
                    else:
                        cp=3
                    if fbs =='Yes':
                        fbs=1
                    else:
                        fbs=0
                    if rest_ecg =='nothing to note':
                        rest_ecg=0
                    elif rest_ecg =='ST-T Wave abnormality':
                        rest_ecg=1
                    else:
                        rest_ecg=2
                    
                    if exang =='Yes':
                        exang=1
                    else:
                        exang=0                    
                    if slope =='Upsloping':
                        slope=0
                    elif slope =='Flatsloping':
                        slope=1
                    else:
                        slope=2
                    if ca =='0':
                        ca=0
                    elif ca =='1':
                        ca=1
                    elif ca =='2':
                        ca=2
                    else:
                        ca=3
                    if thal=='fixed defect':
                        thal=1
                    elif thal =='normal':
                        thal=2
                    else:
                        thal=3


                    pkl_knn_model1=jb.load("heart_model.jb")                    
                    result = pkl_knn_model1.predict(self.st.transform([[age,gender,cp,bp,chol,fbs,rest_ecg,thalach,exang,old_peak,slope,ca,thal]]))
                    # global engine
                    if result== [0]:
                        
                        st.success("This person has not heart disease")
                    else:
                        
                        st.error("This person has heart disease")



            elif task=='EDA':
                st.set_option('deprecation.showPyplotGlobalUse', False)
                heart_df=pd.read_csv("heart dataset after prep/heart_dataset.csv")
                st.header("Exploratory Data Analysis")

                df=pd.DataFrame(heart_df[:20],columns=["age","sex","target"])
                df.hist()
                plt.show()
                st.pyplot()            


                st.line_chart(heart_df[:10])
                # st.area_chart(heart_df)
                # plt.hist(heart_df,bins=20)
                # st.bar_chart(heart_df)
                st.bar_chart(heart_df[:10])

                fig,ax=plt.subplots()
                ax.hist(heart_df[:20],bins=20)
                st.pyplot(fig)

                # hist_data=[df["age"],df["sex"]]
                # group_labels=['age',"sex"]
                # fig=ff.create_distplot(heart_df,group_labels,bin_size=[10,20])
                # st.plotly_chart(fig,user_container_width=True)





            elif task=='Dataset Description':

                st.subheader("Dataset Description")
                st.markdown("""
                            The dataset used in this article is the Cleveland Heart Disease dataset taken from the UCI repository and also available on the kaggle website. The dataset consists of 302 individual’s data. There are 14 columns in the dataset, which are described below.
                            1.  Age: displays the age of the individual.
                            2.  Sex: displays the gender of the individual using the following format :
                                    1 = male
                                    0 = female
                            3.  Chest-pain type: displays the type of chest-pain experienced by the individual using the following format :
                                    0 = typical angina
                                    1 = atypical angina
                                    2 = non — anginal pain
                                    3 = asymptotic
                            4.  Resting Blood Pressure: displays the resting blood pressure value of an individual in mmHg (unit)
                            5.  Serum Cholestrol: displays the serum cholesterol in mg/dl (unit)
                            6.  Fasting Blood Sugar: compares the fasting blood sugar value of an individual with 120mg/dl.
                                    If fasting blood sugar > 120mg/dl then : 1 (true)
                                    else : 0 (false)
                            7.  Resting ECG : displays resting electrocardiographic results
                                    0 = normal
                                    1 = having ST-T wave abnormality
                                    2 = left ventricular hyperthrophy
                            8.  Max heart rate achieved: displays the max heart rate achieved by an individual.
                            9.  Exercise induced angina :
                                    1 = yes
                                    0 = no
                            10. ST depression induced by exercise relative to rest: displays the value which is an integer or float.
                            11. Peak exercise ST segment :
                                    1 = upsloping
                                    2 = flat
                                    3 = downsloping
                            12. Number of major vessels (0–3) colored by flourosopy : displays the value as integer or float.
                            13. Thal : displays the thalassemia :
                                    0 = normal
                                    1 = fixed defect
                                    7= reversible defect
                            14. Diagnosis of heart disease: Displays whether the individual is suffering from heart disease or not: 
                                    0 = absence
                                    1= present.

                    """)
            
        elif selected=="about us":
             st.markdown("AMU")
             image = Image.open('assets/image/logo.png')
             st.image(image, caption='Sunrise by the mountains')
             st.markdown("About me")

             st.markdown(
                        """Amare Abayneh 
                         \n I am student of MSc. in Computer science in 2022GC/2014EC.
                        """)


        elif selected=="contact":
            st.subheader("Contact us ")
            st.markdown("Developer"
                        "\n1. Name: Amare Abayneh" 
                        "\n2. Email: amare0986@gmail.com"
                        "\n3. Phone Number:0986848433")



            
if __name__=="__main__":
    obj=heart_prediction()
    obj.main()

