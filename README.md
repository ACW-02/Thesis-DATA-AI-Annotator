# Thesis-DATA-AI-Annotator
This is my thesis project, building an AI Annotator for Remarks regarding small group member exit reasons. 


paper title : "Automated Text Classification of Small Group Dropout Reasons Using Zero-Shot Large Language Models"

Abstract
This study explains the operational bottlenecks and data privacy risks associated with manually analyzing qualitative free-text responses within a large-scale, metropolitan volunteer-led small group megachurch's. To resolve this, we developed the Automated Semantic Annotation System (ASAS), an automated zero-shot text classification system using Large Language Models (LLMs) with enterprise APIs and Zero Data Retention (ZDR) policies. We evaluated three state-of-the-art models—GPT-5.4-mini, Llama 3.3 70B, and Claude 4.5 Haiku—against a human-annotated ground truth of 2,380 records. Cohen's Kappa was used to test the quality of the ground truth which yielded a moderate agreement coefficient of 0.5315, with discrepancies resolved through structured consensus meetings. A strict JSON schema is implemented and by locking the temperature parameter at 0.0, all models achieved 100% self-consistency.
We evaluate using Cochran’s Q test that resulted in statistically significant performance differences among the models (Q=12.6489,df=2,p=0.001792<0.05). Utilizing McNemar's test with Yates's continuity correction helped confirmed that GPT-5.4-mini significantly outperformed Claude 4.5 Haiku (p=0.0005), achieving the highest accuracy (74.00%) and Macro F1-score (0.69). On the contrary, Llama 3.3 70B demonstrated competitive parity with GPT-5.4-mini (p=0.0673), means a highly viable on-premise alternative to preserve organizational data sovereignty. Deploying the best model into an interactive Streamlit-based web prototype fixes the operational bottleneck, shifting reporting latency from months to minutes. The prototype achieved a 100% score in Black Box testing and a User Acceptance Test (UAT) score of 4.3 out of 5.0 (Excellent).

Keywords—Large Language Models, Text Classification, Zero-shot Learning, Structured Outputs, Hypothesis Significance Testing, Human-in-the-Loop.
