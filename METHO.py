import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Movie Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    df = pd.read_csv(r"C:\Users\LOQ\Desktop\methodologyproject\clean_movies.csv")
except FileNotFoundError:
    st.error("❌ ملف البيانات غير موجود! يرجى التأكد من المسار.")
    st.stop()


if "main_genre" not in df.columns:
    df["main_genre"] = "Unknown"


numeric_cols = ['vote_average', 'popularity', 'vote_count', 'year']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


required_columns = ['year', 'vote_average', 'popularity', 'main_genre']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.warning(f"⚠️ الأعمدة التالية مفقودة: {missing_columns}. بعض الميزات قد لا تعمل.")


def analyze_trends(df):
    """
    تحليل اتجاهات:
    - تطور التقييمات عبر السنوات
    - تطور الشعبية عبر السنوات
    - تغير الأنواع الشائعة
    """
    trends = {}
    
    
    rating_trend = df.groupby('year')['vote_average'].agg(['mean', 'median', 'count']).reset_index()
    
    
    popularity_trend = df.groupby('year')['popularity'].agg(['mean', 'median']).reset_index()
    
    
    genre_trend = df.groupby(['year', 'main_genre']).size().reset_index(name='count')
    top_genre_per_year = genre_trend.loc[genre_trend.groupby('year')['count'].idxmax()]
    
    trends['rating'] = rating_trend
    trends['popularity'] = popularity_trend
    trends['genre'] = top_genre_per_year
    
    return trends

def classify_movies(df):
    """تصنيف الأفلام إلى ناجحة ومتوسطة وفقيرة"""
    df_classified = df.copy()
    
    
    df_classified['success_category'] = pd.cut(
        df_classified['vote_average'],
        bins=[0, 5, 7, 10],
        labels=['Poor 📉', 'Average 📊', 'Successful 🎬']
    )
    
    return df_classified

def simple_recommender(current_movie, df, n=5):
    
    if current_movie not in df['title'].values:
        return pd.DataFrame()
    
   
    genre = df[df['title'] == current_movie]['main_genre'].iloc[0]
    same_genre = df[df['main_genre'] == genre].copy()
    
    
    same_genre = same_genre[same_genre['title'] != current_movie]
    
    if len(same_genre) == 0:
        return df[df['title'] != current_movie].head(n)
    
    
    current_year = df[df['title'] == current_movie]['year'].iloc[0]
    current_rating = df[df['title'] == current_movie]['vote_average'].iloc[0]
    
    same_genre['similarity'] = 1 / (
        1 + 
        0.3 * abs(same_genre['year'] - current_year) +
        2 * abs(same_genre['vote_average'] - current_rating) +
        0.1 * abs(same_genre['popularity'] - df[df['title'] == current_movie]['popularity'].iloc[0])
    )
    
    return same_genre.nlargest(n, 'similarity')[['title', 'year', 'vote_average', 'main_genre', 'popularity']]

 
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #1E40AF 50%, #1D4ED8 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1E90FF;
    }
    .success-box {
        background-color: #2E7D32;
        padding: 10px;
        border-radius: 5px;
        color: white;
        margin: 5px 0;
    }
    .average-box {
        background-color: #F57C00;
        padding: 10px;
        border-radius: 5px;
        color: white;
        margin: 5px 0;
    }
    .poor-box {
        background-color: #C62828;
        padding: 10px;
        border-radius: 5px;
        color: white;
        margin: 5px 0;
    }
    .recommendation-card {
        background-color: #2D3748;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4FD1C5;
    }
</style>
""", unsafe_allow_html=True)


selected = option_menu(
    menu_title=None,
    options=["🏠 Home", "📊 Visualizations", "🔍 Filtering", "🤖 AI Models", "🎯 Recommendations"],
    icons=["house", "bar-chart", "funnel", "robot", "stars"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#1E1E1E"},
        "icon": {"color": "#FFD700", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#2E2E2E",
        },
        "nav-link-selected": {"background-color": "#1E90FF"},
    }
)

# === صفحة Home ===
if selected == "🏠 Home":
    st.markdown('<h1 class="main-header">🎬 Movie Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎥 Total Movies",
            value=f"{len(df):,}",
            delta=f"{len(df[df['year'] == df['year'].max()])} in {int(df['year'].max())}"
        )
    
    with col2:
        avg_vote = df["vote_average"].mean()
        st.metric(
            label="⭐ Average Rating",
            value=f"{avg_vote:.2f}",
            delta=f"±{df['vote_average'].std():.2f} std"
        )
    
    with col3:
        most_popular = df.loc[df["popularity"].idxmax()]
        st.metric(
            label="🔥 Most Popular",
            value=most_popular["title"][:20] + "..." if len(most_popular["title"]) > 20 else most_popular["title"]
        )
    
    with col4:
        avg_popularity = df["popularity"].mean()
        st.metric(
            label="📈 Avg Popularity",
            value=f"{avg_popularity:.2f}",
            delta=f"{df['popularity'].max() - avg_popularity:.2f} from max"
        )
    
    st.subheader("📈 Quick Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top genres
        top_genres = df["main_genre"].value_counts().head(10).reset_index()
        fig_genres = px.bar(
            top_genres,
            x="main_genre",
            y="count",
            title="Top 10 Genres",
            color="count",
            color_continuous_scale="Viridis",
            text="count"
        )
        fig_genres.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_genres, use_container_width=True)
    
    with col2:
        # Year trend
        yearly_stats = df.groupby("year").agg({
            "vote_average": "mean",
            "popularity": "mean"
        }).reset_index()
        
        fig_trend = px.line(
            yearly_stats,
            x="year",
            y=["vote_average", "popularity"],
            title="Yearly Trends",
            markers=True,
            labels={"value": "Score", "variable": "Metric"}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    

    st.subheader("📋 Data Sample")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        sample_size = st.slider("Sample Size", 5, 50, 10)
    
    st.dataframe(
        df.head(sample_size),
        use_container_width=True,
        hide_index=True,
        column_config={
            "title": "Title",
            "year": "Year",
            "vote_average": st.column_config.NumberColumn("Rating", format="%.2f"),
            "popularity": st.column_config.NumberColumn("Popularity", format="%.2f"),
            "main_genre": "Genre"
        }
    )

# === صفحة Visualizations ===
elif selected == "📊 Visualizations":
    st.title("📊 Advanced Visualizations")
    
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Time Series", "Distributions", "Comparisons", "Correlations"]
    )
    
    if viz_type == "Time Series":
        tab1, tab2, tab3 = st.tabs(["Movies per Year", "Ratings Over Time", "Popularity Trend"])
        
        with tab1:
            movies_per_year = df["year"].value_counts().sort_index().reset_index()
            movies_per_year.columns = ["year", "count"]
            fig1 = px.line(
                movies_per_year,
                x="year",
                y="count",
                title="Movies Released per Year",
                markers=True,
                template="plotly_dark",
                line_shape="spline"
            )
            fig1.update_traces(line=dict(width=3))
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            yearly_rating = df.groupby("year")["vote_average"].mean().reset_index()
            fig2 = px.scatter(
                yearly_rating,
                x="year",
                y="vote_average",
                title="Average Rating per Year",
                trendline="lowess",
                template="plotly_dark",
                trendline_color_override="red"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            yearly_popularity = df.groupby("year")["popularity"].mean().reset_index()
            fig3 = px.area(
                yearly_popularity,
                x="year",
                y="popularity",
                title="Popularity Trend Over Years",
                template="plotly_dark"
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    elif viz_type == "Distributions":
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(
                df,
                x="vote_average",
                nbins=30,
                title="Vote Average Distribution",
                template="plotly_dark",
                color_discrete_sequence=['#1E90FF'],
                marginal="box"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.histogram(
                df,
                x="popularity",
                nbins=40,
                title="Popularity Distribution",
                template="plotly_dark",
                color_discrete_sequence=['#FF4500'],
                marginal="violin"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Comparisons":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            top_n = st.slider("Number of Movies", 5, 20, 10)
            sort_by = st.selectbox("Sort by", ["Popularity", "Rating", "Vote Count"])
        
        with col2:
            if sort_by == "Popularity":
                top_movies = df.sort_values(by="popularity", ascending=False).head(top_n)
                y_value = "popularity"
            elif sort_by == "Rating":
                top_movies = df.sort_values(by="vote_average", ascending=False).head(top_n)
                y_value = "vote_average"
            else:
                top_movies = df.sort_values(by="vote_count", ascending=False).head(top_n)
                y_value = "vote_count"
            
            fig = px.bar(
                top_movies,
                x="title",
                y=y_value,
                color="main_genre",
                title=f"Top {top_n} Movies by {sort_by}",
                hover_data=["year", "vote_average", "popularity"],
                template="plotly_dark",
                text=y_value
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlations":
        fig = px.scatter_matrix(
            df,
            dimensions=["vote_average", "popularity", "vote_count", "year"],
            color="main_genre",
            title="Correlation Matrix",
            template="plotly_dark"
        )
        
        fig.update_layout(
            title_font_size=20,
            font_size=8,
            height=900,
            width=1200
        )

        st.plotly_chart(fig, use_container_width=True)


# === صفحة Filtering ===
elif selected == "🔍 Filtering":
    st.title("🔍 Advanced Filtering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # فلترة السنة
        year_range = st.slider(
            "Year Range",
            int(df["year"].min()),
            int(df["year"].max()),
            (int(df["year"].min()), int(df["year"].max()))
        )
    
    with col2:
        # فلترة التقييم
        rating_range = st.slider(
            "Rating Range",
            0.0, 10.0,
            (float(df["vote_average"].min()), float(df["vote_average"].max())),
            step=0.1
        )
    
    with col3:
        # فلترة الشعبية
        popularity_range = st.slider(
            "Popularity Range",
            float(df["popularity"].min()),
            float(df["popularity"].max()),
            (float(df["popularity"].min()), float(df["popularity"].max()))
        )
    
    # فلترة إضافية
    with st.expander("Advanced Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_genres = st.multiselect(
                "Select Genres",
                df["main_genre"].unique(),
                default=None,
                placeholder="All genres"
            )
        
        with col2:
            min_votes = st.number_input(
                "Minimum Votes",
                min_value=0,
                value=0,
                step=100
            )
    
    
    filtered_df = df.copy()
    

    filtered_df = filtered_df[
        (filtered_df["year"] >= year_range[0]) & 
        (filtered_df["year"] <= year_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df["vote_average"] >= rating_range[0]) & 
        (filtered_df["vote_average"] <= rating_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df["popularity"] >= popularity_range[0]) & 
        (filtered_df["popularity"] <= popularity_range[1])
    ]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df["main_genre"].isin(selected_genres)]
    
    if min_votes > 0:
        filtered_df = filtered_df[filtered_df["vote_count"] >= min_votes]
    
    # عرض النتائج
    st.subheader(f"📋 Filtered Results: {len(filtered_df):,} movies found")
    
    if len(filtered_df) > 0:
        # إحصائيات سريعة
        cols = st.columns(4)
        stats = {
            "Avg Rating": filtered_df["vote_average"].mean(),
            "Avg Popularity": filtered_df["popularity"].mean(),
            "Total Votes": filtered_df["vote_count"].sum(),
            "Unique Genres": filtered_df["main_genre"].nunique()
        }
        
        for col, (label, value) in zip(cols, stats.items()):
            with col:
                st.metric(label, f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
        
        # عرض البيانات
        st.dataframe(
            filtered_df.sort_values("popularity", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "title": st.column_config.TextColumn("Title", width="large"),
                "year": st.column_config.NumberColumn("Year", format="%d"),
                "vote_average": st.column_config.NumberColumn("Rating", format="%.2f"),
                "popularity": st.column_config.NumberColumn("Popularity", format="%.2f"),
                "main_genre": "Genre"
            }
        )
        
        # خيار التصدير
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_movies.csv",
            mime="text/csv",
            type="primary"
        )
        
        # مخطط سريع للبيانات المفلترة
        if len(filtered_df) > 1:
            st.subheader("📊 Filtered Data Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.scatter(
                    filtered_df,
                    x="vote_average",
                    y="popularity",
                    color="main_genre",
                    size="vote_count",
                    hover_name="title",
                    title="Rating vs Popularity (Filtered)"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                genre_counts = filtered_df['main_genre'].value_counts().reset_index()
                fig2 = px.pie(
                    genre_counts,
                    names='main_genre',
                    values='count',
                    title="Genre Distribution (Filtered)",
                    hole=0.4
                )
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ No movies match your filters. Try adjusting them.")

# === صفحة AI Models ===
elif selected == "🤖 AI Models":
    st.title("🤖 AI Models & Trend Analysis")
    
    # إنشاء تبويبات داخل الصفحة
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "🎯 Success Classifier", "🔮 Rating Predictor"])
    
    with tab1:
        st.header("📈 Trend Analysis")
        
        
        trends = analyze_trends(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Rating Trend")
            fig1 = px.line(
                trends['rating'],
                x='year',
                y='mean',
                title='Average Movie Ratings Over Years',
                markers=True,
                line_shape='spline',
                color_discrete_sequence=['#1E90FF']
            )
            fig1.update_layout(
                xaxis_title="Year",
                yaxis_title="Average Rating",
                hovermode='x unified'
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # عرض بيانات التقييمات
            with st.expander("📊 View Rating Trend Data"):
                st.dataframe(trends['rating'].round(3), use_container_width=True)
        
        with col2:
            st.subheader("Popularity Trend")
            fig2 = px.line(
                trends['popularity'],
                x='year',
                y='mean',
                title='Popularity Over Years',
                markers=True,
                line_shape='spline',
                color_discrete_sequence=['#FF6B6B']
            )
            fig2.update_layout(
                xaxis_title="Year",
                yaxis_title="Average Popularity"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # عرض بيانات الشعبية
            with st.expander("📊 View Popularity Trend Data"):
                st.dataframe(trends['popularity'].round(3), use_container_width=True)
        
        # مخطط الأنواع الشائعة
        st.subheader("Most Popular Genres Each Year")
        
        fig3 = px.bar(
            trends['genre'],
            x='year',
            y='count',
            color='main_genre',
            title='Dominant Movie Genres by Year',
            hover_data=['main_genre'],
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig3.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Movies",
            showlegend=True,
            xaxis_tickangle=45
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # إحصائيات سريعة
        st.subheader("📊 Quick Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # أفضل سنة من حيث التقييم
            best_year_rating = trends['rating'].loc[trends['rating']['mean'].idxmax()]
            st.metric(
                "⭐ Best Rated Year", 
                f"{int(best_year_rating['year'])}",
                f"{best_year_rating['mean']:.2f} avg"
            )
        
        with col2:
            # أفضل سنة من حيث الشعبية
            best_year_pop = trends['popularity'].loc[trends['popularity']['mean'].idxmax()]
            st.metric(
                "🔥 Most Popular Year", 
                f"{int(best_year_pop['year'])}",
                f"{best_year_pop['mean']:.2f} avg"
            )
        
        with col3:
            # أكثر نوع تكرراً
            most_common_genre = trends['genre']['main_genre'].mode()[0]
            genre_count = len(trends['genre'][trends['genre']['main_genre'] == most_common_genre])
            st.metric(
                "🏆 Most Common Genre", 
                most_common_genre,
                f"{genre_count} times"
            )
    
    with tab2:
        st.header("🎯 Success Classifier")
        
        # تطبيق التصنيف
        df_classified = classify_movies(df)
        
        # عرض النتائج
        col1, col2, col3 = st.columns(3)
        
        with col1:
            successful = len(df_classified[df_classified['success_category'] == 'Successful 🎬'])
            st.metric("Successful Movies", f"{successful:,}")
        
        with col2:
            average = len(df_classified[df_classified['success_category'] == 'Average 📊'])
            st.metric("Average Movies", f"{average:,}")
        
        with col3:
            poor = len(df_classified[df_classified['success_category'] == 'Poor 📉'])
            st.metric("Poor Movies", f"{poor:,}")
        
        # مخطط دائري
        fig = px.pie(
            df_classified,
            names='success_category',
            title='Movie Success Distribution',
            hole=0.3,
            color_discrete_sequence=['#00C851', '#FFBB33', '#FF4444']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # مخطط شريطي للأنواع حسب النجاح
        st.subheader("Success by Genre")
        
        success_by_genre = df_classified.groupby(['main_genre', 'success_category']).size().reset_index(name='count')
        fig2 = px.bar(
            success_by_genre,
            x='main_genre',
            y='count',
            color='success_category',
            title='Movie Success by Genre',
            barmode='group',
            color_discrete_sequence=['#FF4444', '#FFBB33', '#00C851']
        )
        fig2.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
        
        # عرض عينة من البيانات
        st.subheader("Sample Classified Movies")
        
        category_filter = st.selectbox(
            "Filter by category:",
            ["All", "Successful 🎬", "Average 📊", "Poor 📉"]
        )
        
        display_df = df_classified.copy()
        if category_filter != "All":
            display_df = display_df[display_df['success_category'] == category_filter]
        
        st.dataframe(
            display_df[['title', 'year', 'vote_average', 'popularity', 'success_category']]
            .sort_values('vote_average', ascending=False)
            .head(20),
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        st.header("🔮 Rating Predictor")
        
        st.info("Predict a movie's rating based on its features")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
        
            st.subheader("Movie Features")
            
            col_year, col_pop, col_votes = st.columns(3)
            
            with col_year:
                year = st.slider("Release Year", 
                               int(df['year'].min()), 
                               int(df['year'].max()), 
                               2020)
            
            with col_pop:
                popularity = st.slider("Expected Popularity", 
                                     float(df['popularity'].min()), 
                                     float(df['popularity'].max()), 
                                     float(df['popularity'].median()))
            
            with col_votes:
                vote_count = st.slider("Expected Vote Count", 
                                     int(df['vote_count'].min()), 
                                     int(df['vote_count'].max()), 
                                     int(df['vote_count'].median()))
            
            # اختيار النوع
            genre = st.selectbox("Movie Genre", df['main_genre'].unique())
        
        with col2:
            st.subheader("Model Info")
            st.markdown("""
            **Algorithm:** Linear Regression
            **Features used:**
            - Release Year
            - Popularity
            - Vote Count
            - Genre
            """)
        
        if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
            try:
                # تحضير البيانات للنموذج
                le = LabelEncoder()
                df_model = df.copy()
                df_model['genre_encoded'] = le.fit_transform(df_model['main_genre'])
                
                #ي
                X = df_model[['year', 'popularity', 'vote_count', 'genre_encoded']].fillna(0)
                y = df_model['vote_average']
                
                model = LinearRegression()
                model.fit(X, y)
                
                # تحويل المدخلات
                if genre in le.classes_:
                    genre_encoded = le.transform([genre])[0]
                else:
                    genre_encoded = 0
                    st.warning(f"Genre '{genre}' not in training data. Using default encoding.")
                
                # التنبؤ
                input_features = [[year, popularity, vote_count, genre_encoded]]
                predicted_rating = model.predict(input_features)[0]
                
                # التأكد من أن التقييم بين 0 و 10
                predicted_rating = max(0, min(10, predicted_rating))
                
                # عرض النتيجة
                st.success(f"""
                ## 📊 Prediction Results
                
                **Predicted Rating:** ⭐ **{predicted_rating:.2f}** / 10
                
                **Features used:**
                - Year: {year}
                - Popularity: {popularity:.2f}
                - Vote Count: {vote_count:,}
                - Genre: {genre}
                """)
                
                # عرض شريط التقييم
                st.progress(int(predicted_rating * 10))
                
                # مقارنة مع المتوسط
                avg_rating = df['vote_average'].mean()
                difference = predicted_rating - avg_rating
                
                if difference > 1:
                    st.info(f"📈 This movie is predicted to be **{difference:.2f} points** above average!")
                elif difference < -1:
                    st.warning(f"📉 This movie is predicted to be **{abs(difference):.2f} points** below average.")
                else:
                    st.info(f"📊 This movie is predicted to be around average.")
                    
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                # نموذج بديل مبسط
                st.info("Using simplified prediction model...")
                
                # محاكاة تنبؤ مبسط
                base_rating = 6.0
                year_effect = (year - 2000) * 0.02
                popularity_effect = np.log(popularity + 1) * 0.5
                votes_effect = np.log(vote_count + 1) * 0.3
                
                predicted_rating = min(10, max(0, 
                    base_rating + year_effect + popularity_effect + votes_effect
                ))
                
                st.success(f"""
                ## 📊 Simplified Prediction
                
                **Predicted Rating:** ⭐ **{predicted_rating:.2f}** / 10
                """)
                st.progress(int(predicted_rating * 10))

# === صفحة Recommendations ===
elif selected == "🎯 Recommendations":
    st.title("🎯 Movie Recommendation System")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Find Similar Movies")
        
        # اختيار فيلم
        movie_list = sorted(df['title'].unique().tolist())
        selected_movie = st.selectbox("Select a movie you like:", movie_list, index=0)
        
        n_recommendations = st.slider("Number of recommendations:", 3, 10, 5)
        
        if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
            recommendations = simple_recommender(selected_movie, df, n_recommendations)
            
            if len(recommendations) > 0:
                st.success(f"🎬 Movies similar to **{selected_movie}**:")
                
                for idx, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>🎬 {row['title']} ({int(row['year'])})</h4>
                            <p>⭐ <b>{row['vote_average']:.1f}</b>/10 | 📊 {row['popularity']:.1f} popularity</p>
                            <p>🏷️ <i>{row['main_genre']}</i></p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try another movie.")
    
    with col2:
        st.subheader("Movie Details")
        
        if 'selected_movie' in locals() or 'selected_movie' in globals():
            movie_data = df[df['title'] == selected_movie]
            
            if len(movie_data) > 0:
                movie = movie_data.iloc[0]
                
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.metric("⭐ Rating", f"{movie['vote_average']:.1f}/10")
                    st.metric("📅 Year", int(movie['year']))
                
                with col_info2:
                    st.metric("📊 Popularity", f"{movie['popularity']:.1f}")
                    st.metric("🏷️ Genre", movie['main_genre'])
                
                # مخطط مقارنة مع المتوسطات
                avg_rating = df['vote_average'].mean()
                avg_popularity = df['popularity'].mean()
                
                fig = px.bar(
                    x=['Rating', 'Popularity'],
                    y=[movie['vote_average'], movie['popularity']],
                    title=f"{selected_movie} vs Averages",
                    labels={'x': 'Metric', 'y': 'Value'},
                    color=['Rating', 'Popularity'],
                    color_discrete_sequence=['#1E90FF', '#FF6B6B']
                )
                
                # إضافة خطوط المتوسطات
                fig.add_hline(y=avg_rating, line_dash="dash", line_color="blue", 
                            annotation_text=f"Avg Rating: {avg_rating:.1f}")
                fig.add_hline(y=avg_popularity, line_dash="dash", line_color="red",
                            annotation_text=f"Avg Popularity: {avg_popularity:.1f}")
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔍 Quick Recommendations by Genre")
        
        genre_list = sorted(df['main_genre'].unique().tolist())
        selected_genre = st.selectbox("Select genre:", genre_list)
        
        top_movies_genre = df[df['main_genre'] == selected_genre]\
            .sort_values('vote_average', ascending=False)\
            .head(5)
        
        for idx, row in top_movies_genre.iterrows():
            with st.expander(f"{row['title']} ({int(row['year'])})"):
                st.write(f"⭐ **{row['vote_average']:.1f}/10**")
                st.write(f"📊 Popularity: {row['popularity']:.1f}")
                st.write(f"🗳️ Votes: {row['vote_count']:,}")

# === Footer ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h3>🎬 Movie Analytics Dashboard</h3>
        <p>Built with ❤️ using Streamlit, Plotly & Pandas | Data Source: TMDB</p>
        <p style='font-size: 0.8rem; color: #999;'>Total Movies: {:,} | Average Rating: {:.2f}/10</p>
    </div>
    """.format(len(df), df['vote_average'].mean()),
    unsafe_allow_html=True
)