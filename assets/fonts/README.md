# 한글 폰트(배포용)

GitHub/Streamlit Cloud(리눅스)로 배포하면 Windows 기본 한글 폰트(맑은 고딕)가 없어서,
`matplotlib`/`wordcloud` 이미지에서 한글이 □□□로 깨질 수 있습니다.

이 폴더에 **TTF 폰트 파일을 추가**하면 앱이 이를 우선 사용하도록 되어 있습니다.

## 자동으로 폰트 받기(추천)

아래 스크립트를 실행하면 `assets/fonts/`에 Noto Sans KR TTF를 자동으로 다운로드/추출합니다.

```bash
python scripts/download_fonts.py
```

권장 파일명(둘 중 하나 세트만 있어도 됩니다):

- `NotoSansKR-Regular.ttf`
- `NotoSansKR-Bold.ttf` (선택)

또는

- `NanumGothic.ttf`
- `NanumGothicBold.ttf` (선택)

추가 후 배포하면 연관성 그래프/워드클라우드의 한글이 정상 출력됩니다.


