import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def load_data(csv_path: str = "StudentsPerformance.csv") -> pd.DataFrame:
    """
    CSV 파일을 읽어오는 함수.
    에러가 나면 Streamlit 화면에 메시지를 보여주고 앱을 멈춥니다.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(
            f"`{csv_path}` 파일을 찾을 수 없습니다.\n\n"
            f"- 이 앱이 있는 폴더(리포지토리 루트)에 **{csv_path}** 파일을 넣어주세요.\n"
            f"- GitHub에 푸시한 뒤 Streamlit Cloud를 다시 배포해 주세요."
        )
        st.stop()
    except Exception as e:
        st.error("CSV 파일을 읽는 중 예기치 못한 오류가 발생했습니다.")
        st.exception(e)
        st.stop()

    return df


def check_columns(df: pd.DataFrame):
    """
    필요한 컬럼이 다 있는지 확인하고,
    없으면 화면에 알려주고 앱을 멈춥니다.
    """
    required_cols = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "math score",
        "reading score",
        "writing score",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(
            "데이터에 필요한 컬럼이 부족합니다.\n\n"
            f"**필요한 컬럼:** {required_cols}\n\n"
            f"**누락된 컬럼:** {missing}"
        )
        st.write("현재 CSV의 컬럼 목록은 다음과 같습니다:")
        st.write(list(df.columns))
        st.stop()


def train_model(df: pd.DataFrame, target_col: str):
    """
    랜덤 포레스트 회귀모델을 학습해서 Pipeline 형태로 반환.
    - 입력 특징: gender, race/ethnicity, parental level of education,
                 lunch, test preparation course
    - 타깃: target_col (math score / reading score / writing score 중 하나)
    """
    feature_cols = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ]

    X = df[feature_cols]
    y = df[target_col]

    # 범주형 변수 리스트
    categorical_features = feature_cols

    # One-Hot Encoding + Random Forest 파이프라인
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    try:
        pipe.fit(X, y)
    except Exception as e:
        st.error("모델을 학습하는 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()

    return pipe


def main():
    st.set_page_config(
        page_title="학생 성적 예측기 (Random Forest)",
        page_icon="📊",
        layout="centered",
    )

    st.title("📊 학생 성적 예측기")
    st.write(
        """
        성별, 인종, 부모 최종학력, 점심 유형, 시험 준비 코스 수강 여부를 입력하면  
        **랜덤 포레스트 회귀(Random Forest Regressor)** 모델로  
        선택한 과목의 점수를 예측해주는 데모입니다.
        """
    )

    # 1. 데이터 로드
    df = load_data()

    # 2. 컬럼 체크
    check_columns(df)

    # 3. 예측할 과목 선택
    st.sidebar.header("⚙️ 설정")
    target_col = st.sidebar.selectbox(
        "예측할 과목을 선택하세요",
        options=[
            "math score",
            "reading score",
            "writing score",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("모델: RandomForestRegressor (데모용 설정)")

    # 4. 입력 UI 만들기
    st.subheader("1️⃣ 학생 정보 입력")

    # 각 범주형 변수의 선택지는 데이터에서 자동으로 가져옴
    gender_options = sorted(df["gender"].dropna().unique())
    race_options = sorted(df["race/ethnicity"].dropna().unique())
    parent_edu_options = sorted(df["parental level of education"].dropna().unique())
    lunch_options = sorted(df["lunch"].dropna().unique())
    prep_options = sorted(df["test preparation course"].dropna().unique())

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("성별 (gender)", gender_options)
        race = st.selectbox("인종/민족 (race/ethnicity)", race_options)
        lunch = st.selectbox("점심 유형 (lunch)", lunch_options)

    with col2:
        parental_edu = st.selectbox(
            "부모 최종학력 (parental level of education)", parent_edu_options
        )
        test_prep = st.selectbox(
            "시험 준비 코스 (test preparation course)", prep_options
        )

    # 5. 예측 버튼
    st.subheader("2️⃣ 예측 실행")
    if st.button("예측하기 🚀"):
        # 입력값을 DataFrame으로 만들어 모델에 넣기
        input_df = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "race/ethnicity": race,
                    "parental level of education": parental_edu,
                    "lunch": lunch,
                    "test preparation course": test_prep,
                }
            ]
        )

        # 버튼을 눌렀을 때마다 간단히 모델을 한 번 학습 (데모이므로 OK)
        model = train_model(df, target_col)

        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error("예측을 수행하는 중 오류가 발생했습니다.")
            st.exception(e)
            st.stop()

        st.success(f"예측된 **{target_col}** 는 약 **{pred:.2f} 점** 입니다.")

        with st.expander("⚗️ 사용된 입력값 보기"):
            st.write(input_df)

    # 6. 데이터 미리보기
    st.markdown("---")
    st.subheader("3️⃣ 학습에 사용된 데이터 (미리보기)")
    st.dataframe(df.head())


if __name__ == "__main__":
    main()
