import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\u0600-\u06FF\w\s-]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens_stemmed = [stemmer.stem(word) for word in tokens]
    # tokens_lemmatized = [lemmatizer.lemmatize(word) for word in tokens_stemmed]
    return ' '.join(tokens_stemmed)


def search(query, descriptions, indices):
    

    query = preprocess_text(query)
    if not query.strip():
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_scores = similarity_scores[ranked_indices]

    results = [(indices[i], ranked_scores[idx]) 
               for idx, i in enumerate(ranked_indices) 
               if ranked_scores[idx] > 0]
    return results

podcasts_online = [
    {
        "title": "رمضان",
        "description": """رمضان ليس مجرد شهر للصيام، بل هو فرصة للتقرب إلى الله وتجديد الإيمان...
الصيام في رمضان يعزز من تقوى الله ويعلم المسلم الصبر والاحتساب...
النية الصادقة هي أساس قبول الأعمال...""",
        "url": "https://open.spotify.com/show/5og74haQRULrM4UqbS8lMB?si=6d4c37b3832a462e"
    },
    {
        "title": "هكسب اي لو التزمت",
        "description": """الالتزام بتعاليم الدين يُعدّ طريقًا لتحقيق النجاح الحقيقي في الدنيا والآخرة...
التدين يمنح الإنسان طمأنينة وسلامًا داخليًا يعينه على مواجهة تحديات الحياة...""",
        "url": "https://open.spotify.com/show/5og74haQRULrM4UqbS8lMB?si=6d4c37b3832a462e"
    },
    {
        "title": "فهم أوسع عن حقيقة الرزق",
        "description": """الرزق لا يقتصر على المال فقط، بل يشمل الصحة، والعلم، والرضا، والعلاقات الطيبة...
التوكل الحقيقي يعني السعي مع الاعتماد القلبي على الله في تحقيق الرزق...""",
        "url": "https://open.spotify.com/show/5og74haQRULrM4UqbS8lMB?si=6d4c37b3832a462e"
    },
    {
        "title": "الأخلاق والحياء",
        "description": """الحياء جزء من الإيمان ويعبر عن التقوى واحترام الحدود...
الحياء مع الله يتضمن الخوف من المعاصي والالتزام بالطاعات...""",
        "url": "https://open.spotify.com/show/5og74haQRULrM4UqbS8lMB?si=6d4c37b3832a462e"
    },
    {
        "title": "الشخصية القوية",
        "description": """الشخصية القوية تعني الثقة بالنفس، الاتزان العاطفي، والقدرة على اتخاذ القرارات بثبات...
الشخصية القوية ليست مرادفًا للتسلط أو السيطرة، بل هي تعني المصداقية والاتزان...
التعلم من الأخطاء يُعتبر أساسًا لبناء شخصية قوية ومتجددة...""",
        "url": "https://open.spotify.com/episode/1vhogwYSMaT9bDmgVAOJgp?si=fcad4ea921bf4f7a"
    },
    {
        "title": "The Daily Listening English",
        "description": "In this story, we practice listening to English as we follow the journey of a man who discovers a glowing black rock in the forest. The unusual events that unfold, such as electronics malfunctioning and strange energy around the rock, will help you improve your listening skills while engaging with an intriguing mystery,Successful relationships help individuals grow and develop personally.,Open communication and recognizing each other's feelings are key to maintaining a healthy relationship",
        "url": "https://open.spotify.com/show/1PFrNV9rldXOQzveGNm7Ll?si=32df7096096b45eb"
    },
        {
        "title": "Love in Long-Term Relationships",
        "description": "Love in long-term relationships does not suddenly disappear but fades due to routine and daily life pressures,Successful relationships help individuals grow and develop personally.",
        "url": "https://podcasts.apple.com/eg/podcast/relationships-2-0-keeping-love-alive/id1028908750?i=1000705268695"
    },        {
        "title": "ما سبب الحملات المستمرة على كريستيانو رونالدو",
        "description": "تناقش الحلقة الضغوط الإعلامية والجماهيرية التي يتعرض لها كريستيانو رونالدو رغم مكانته التاريخية، بالإضافة إلى تحليل تراجع كرة القدم الإيطالية وغياب المنتخب البلجيكي عن تحقيق البطولات.",
        "url": "https://open.spotify.com/episode/2THaJAGHf64ylWM61bjGNE?si=JAFiW0pnTIONEt3cxLyy9Q"
    },
            {
        "title": "هل تحسم الخبرة قمة الأهلي والهلال",
        "description": "تناقش هذه الحلقة المواجهة المرتقبة بين الهلال والأهلي في نصف نهائي دوري أبطال آسيا، بالإضافة إلى قضايا مثل تراجع المنافسة بين أندية شرق وغرب آسيا ومشكلة التهديف في نادي النصر.",
        "url": "https://open.spotify.com/episode/1QdmSxwllPrtKXOuBRqJZX?si=UN0_wbEbQKSRyO85SuLY6w"
    },
            {
        "title": " كريم بنزيما لاعبًا في الاتحاد",
        "description": "حلقة تسلط الضوء على صفقة انتقال كريم بنزيما إلى نادي الاتحاد السعودي، وتناقش أبعادها الرياضية والتسويقية، إضافة إلى انعكاسها على المنافسة في الدوري السعودي.",
        "url": "https://open.spotify.com/episode/53EUbQ1pJ30s2V9rvrioC3?si=gx261kHXR7ury6CCKTcmDQ"
    },
            {
        "title":"ردة الفعل الأوروبية على استضافة السعودية لكأس العالم",
        "description": "حلقة تستعرض تفاصيل ملف استضافة السعودية لكأس العالم 2034، والانتقادات الغربية التي واجهتها، بالإضافة إلى الحديث عن مستقبل الرياضة السعودية ومباريات الأندية في الدوري المحلي.",
        "url": "https://open.spotify.com/episode/4osebAi7AVSoJglM78K9ki?si=706e429b2e2549d3"
    },
                {
        "title": " ايه المشكله في بدايه الالتزام",
        "description":"نقاش عن أسباب تكرار الفتور بعد بداية الالتزام، الفرق بين الالتزام الحقيقي والشكلي، وكيفية الاستمرار رغم التحديات والالتزام مش بس لبس أو مظهر، بل تغيير داخلي مستمر في فرق بين التدين الشكلي والوعي الديني الحقيقي",
        "url": "https://open.spotify.com/episode/4Kk1hOzuiiecS9PnE5d5FN?si=861a2e5ae8ec4a40"
    },{
        "title": "ايه المشكله في التنمر",               
        "description": "تعريف التنمر وأنواعه، الفرق بين المزاح والتنمر، آثاره النفسية، وكيفية التعامل معه من منظور ديني ومجتمعي آثار التنمر على الضحية:انخفاض الثقة بالنفس الانعزال الاجتماعي والاكتئاب",
        "url": "https://open.spotify.com/episode/2opKS2OvJv5xKGA6SlqdDx?si=27ffcfaabcd549e3"
    },
                        {
        "title": "ايه المشكله في الانضباط وضياع الوقت",               
        "description": "ما هو الانضباط الذاتي؟ وأسباب ضياع الوقت؟ مع نصائح عملية لتعزيز التركيز والانضباط أسباب ضياع الوقت:التسويف وتأجيل المهام.الانشغال بالمشتتات مثل وسائل التواصل الاجتماعيعدم وجود خطة واضحة أو أهداف محددة​",
        "url": "https://open.spotify.com/episode/1EvCPKXvy3ItFXiYtWCBLF?si=1d892a9a16b24d3d"
    },
                            {
        "title": "ايه المشكله في الرياء",               
        "description": "ما هو الرياء؟ وكيف يؤثر على النية والعمل؟ مع نصائح للوقاية منه وتحقيق الإخلاص الرياء هو القيام بالأعمال الصالحة بقصد نيل إعجاب الناس وليس لوجه الله",
        "url": "https://open.spotify.com/episode/16yAx99xCuBYKGke4rHNYT?si=fc319ae0d7354010"
    },
        {
        "title": "  ايه المشكله في الرزق",
        "description": "الرزق لا يقتصر على المال، بل يشمل الصحة والعلاقات وغيرها، مع مناقشة مفاهيم القناعة والرضا الله هو الرازق، وكل إنسان مكتوب له رزقه منذ ولادته و ضرورة الأخذ بالأسباب والعمل الجاد",
        "url": "https://open.spotify.com/episode/61OJVEFCgvE9Bckz75mk9q?si=cde952b53eb740c5",
        
    },        {
        "title": "  ايه المشكله في الترندات",
        "description":" الترندات هي المواضيع أو الظواهر التي تكتسب شهرة وانتشارًا واسعًا في فترة زمنية قصيرة الآثار الإيجابية للترندات: نشر الوعي حول قضايا مهمة.تعزيز التفاعل والتواصل بين الأفراد الآثار السلبية للترندات نشر معلومات خاطئة أو مضلله والضغط النفسي للمشاركة دون قناعه",
        "url": "https://open.spotify.com/episode/6lOob7itwftHqzXrCC2AXS?si=01f88b5deb14483c",
    },
            {
        "title": "  ايه المشكله في النجاح",
        "description": "ما هو النجاح الحقيقي؟ وكيف نحققه دون أن نفقد توازننا أو قيمنا؟ النجاح ليس مجرد تحقيق الأهداف المادية، بل يشمل أيضًا السلام الداخلي والرضا الشخصي تفاوت مفاهيم النجاح بين الأفراد بناءً على القيم والطموحات الشخصية.",
        "url": "https://open.spotify.com/episode/5UzDpmZutJoopxgkdFARju?si=9c602503f2b84904",
    },
{
    "title": "  ايه المشكله في النسيان",
    "description": "نظرة دينية وعلمية على النسيان، أسبابه، آثاره، وكيفية التعامل معه بمرونة. النسيان هو فقدان الذاكرة أو عدم التذكر، وهو أمر طبيعي يحدث للجميع. في الإسلام، يُعتبر النسيان من الله رحمة بالعباد.",
    "url": "https://open.spotify.com/episode/4WnXypIXsPObw5V6MnS4J7?si=0a3ca538fcee4581",
},

{
    "title": "  ايه المشكله في الموت",
    "description": "الموت من منظور ديني، كيف نتعامل مع الفقد؟ وكيف نستعد له؟ الموت هو انتقال الإنسان من الحياة الدنيا إلى الآخرة. في الإسلام، يُعتبر الموت جزءًا من قضاء الله وقدره.",
    "url": "https://open.spotify.com/episode/4WnXypIXsPObw5V6MnS4J7?si=0a3ca538fcee4581",
},
{
    "title": "قوة المشاعر",
    "description": "في أعماق كل تجربة نعيشها تكمن مشاعر تُشكّل فهمنا لذواتنا وتوجّه تصرفاتنا. لكل إنسان مشاعر تنبع من تجاربه الخاصة، وإدراك هذه المشاعر وتسميتها هو خطوة أساسية في رحلة الوعي الذاتي هذا الوعي لا يساعدنا فقط على فهم أنفسنا بشكل أعمق، بل يلعب دورًا محوريًا في جودة علاقاتنا مع الآخرين  أهم جداً أساساً التكافه يمنحك ثقة بالنفس عالية وقوة",
    "url": "https://youtu.be/mHbfg_arzQk?si=9JruxvgP3ZW_KHDZ",
},
        {
    "title": " هدوء",
    "description": "كيفية تأثير الكلمات والمفاهيم المتكررة على الذات والعلاقات. من التحديات اليومية إلى التفاعلات مع الأشخاص المحيطين، يعبر هاشم عن تجربته الشخصية في فهم ذاته والتعامل مع التوقعات المختلفة. تبدأ قصته بتفاصيل تبدو بسيطة لكنها تحمل عمقًا في فهم النفس والعلاقات الإنسانية، مشيرًا إلى أهمية التوازن بين الفردانية والتواصل الاجتماعي. تعكس الكلمات التي يشاركها كيف أن الحياة مليئة بالتجارب التي تشكل الفهم الداخلي للعالم وكيف يمكن للثنائيات والعلاقات أن تساعد في إعادة صياغة المعاني الحقيقية للحياة.",
    "url": "https://youtu.be/8XIrG-KkHmg?si=EY0rqY67aptbkUzf",
},
        {
    "title": " القوة النفسيه",
    "description":"الهروب والقدرة على التعامل مع الألم في مواجهة التحديات، يتعامل بعض الناس مع مشاعرهم بتجنب مباشر للألم والمعاناة، وهؤلاء مثل المسكينات السريعة، التي قد تبدو سريعة في تخفيف الألم، لكن على المدى البعيد، يتحول بناؤهم النفسي إلى هش وضعيف، غير قادر على التحمل",
    "url": "https://youtu.be/-i1uF3pa0oI?si=veV0TEMSwoKlJhdM",
},
{
    "title": "انواع الاكتئاب",
    "description": "الاكتئاب مش مجرد شعور بالحزن أو حالة مؤقتة بتمر، لكنه اضطراب نفسي معقد بيأثر على طريقة التفكير، الإحساس، والتعامل مع الحياة وأن الاكتئاب لا يقتصر على شكل واحد، بل يشمل عدة أنواع مثل الاكتئاب الجسيم، المزمن، الموسمي، واكتئاب ما بعد الصدمات ",
    "url": "https://podcasts.apple.com/eg/podcast/%D8%A7%D9%84%D8%A7%D9%83%D8%AA%D8%A6%D8%A7%D8%A8/id1533707735?i=1000492855907"
},
{
    "title": " Sleep Cycle ",
    "description": " the science of sleep, offering insights into how factors like brain stimulation, temperature regulation, and sound therapy can enhance sleep quality and improve overall well-being.",
    "url": "https://podcasts.apple.com/eg/podcast/sleep-enhancement-with-prof-matthew-walker-pt-1/id1723189267?i=1000639824113"
},
{
    "title": "بزنس بالعربي ",
    "description": "شهدت المنتجات المحلية في مصر تحولًا ملحوظًا، حيث أصبح المستهلكون في السوق المصري يفضلون العلامات التجارية المحلية أكثر من أي وقت مضى و "      ,  "url": "https://podcasts.apple.com/eg/podcast/%D8%A8%D8%B2%D9%86%D8%B3-%D8%A8%D8%A7%D9%84%D8%B9%D8%B1%D8%A8%D9%8A-business-%D8%A8%D8%A7%D9%84%D8%B9%D8%B1%D8%A8%D9%89/id1490825968"
},{
    "title": "صعب اني اسامح",
    "description": "مفهوم التسامح كأداة نفسية مهمة في حياتنا اليومية، وكيف يمكن أن يكون له تأثير إيجابي في تجاوز الأزمات العاطفية والعلاقات المتوترة. التحدي الأكبر يكمن في كيف أن بعض الأشخاص يجدون صعوبة في المسامحة بعد تعرضهم للألم أو الخيانة، ويعتقدون أن التسامح قد يُعتبر علامة على الضعف أو التنازل. ",
    "url": "https://podcasts.apple.com/eg/podcast/%D8%B5%D8%B9%D8%A8-%D8%A7%D9%86%D9%8A-%D8%A3%D8%B3%D8%A7%D9%85%D8%AD/id1777806076?i=1000700655261"
},
{
    "title": "سوالف تغذية",
    "description": "بناء نظام غذائي متوازن يشمل جميع المجموعات الغذائية المهمة مثل البروتينات، الكربوهيدرات، الدهون، الفيتامينات والمعادن ويكون دور التمارين الرياضية في تعزيز الصحة العامة أهمية التغذية قبل وبعد التمرين" ,
   "url": "https://podcasts.apple.com/eg/podcast/7-%D8%A7%D8%B3%D8%A6%D9%84%D8%A9-%D9%88-%D8%A7%D8%AC%D9%88%D8%A8%D8%A9-%D9%85%D8%AA%D9%86%D9%88%D8%B9%D8%A9/id1567442005?i=1000635690697"
}
]

podcast_titles = []
podcast_descriptions = []
podcast_urls = []

for podcast in podcasts_online:
    full_text = f"{podcast['title']} {podcast['description']}"
    cleaned = preprocess_text(full_text)
    if cleaned.strip():
        podcast_titles.append(podcast['title'])
        podcast_descriptions.append(cleaned)
        podcast_urls.append(podcast['url'])

# =============================================================================
import streamlit as st

st.set_page_config(page_title="Podcast Search engine",page_icon="🔎")
st.title(" Podcast Search Engine")
st.markdown(
   """
    <style>
    .stApp {

        background-image: url("https://img.freepik.com/premium-photo/studio-podcast-microphone-dark-background_162008-316.jpg?w=1380");
        background-size: 110% 110%;
        background-position: 110% 50% ;
        opacity: 0.8;
        background-repeat: no-repeat;
    }

    
    header {visibility: hidden;}

    </style>
    """,
    unsafe_allow_html=True
)
query = st.text_input("", placeholder=" أبهرني عايز تشوف اي ")
query = query.strip()
if query:
    
    results = search(query, podcast_descriptions, list(range(len(podcast_titles))))

    if results:
        st.success(f"تم العثور على {len(results)} نتيجة:")
        
        relevant_flags = []
        for idx, score in results:
            st.subheader(f"🎧 Title:  {podcast_titles[idx]}")
            st.markdown(f"🔗 **Link** :  {podcast_urls[idx]}")
            st.markdown(f"**📝 Description:**  {podcasts_online[idx]['description']}")
            st.markdown(f"✅ **Similarity Score**:  {round(score * 100, 2)}٪")
            
            marked = st.checkbox("Mark as relevant", key=podcast_titles[idx])
            relevant_flags.append(marked)
            st.divider()
            # st.markdown("---")
        relevant_count = sum(relevant_flags)
        ratio = relevant_count / len(results)
        st.info(f"📊 Precision: {round(ratio, 2)}")
    else:
        st.warning("لم يتم العثور على نتائج مطابقة.")
else:
    st.info("يرجى إدخال عبارة للبحث.")