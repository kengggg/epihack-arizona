## EpiHack 2026: User data analysis information products

<p><img src="https://github.com/kengggg/epihack-arizona/blob/main/analysis/figs/epihack-community-health.png" width=900></p>

***

Patipat Susumpao, Opendream Co. Ltd. <br>
Carlos Lizárraga-Celaya, University of Arizona <br>
<br>
[**EpiHack Arizona 2026**](https://arizona.epihack.org/)

***


**All mockup dashboards created using a small data sample `epihack-mini.db`, provided by Patipat Susumpao (Keng)**
 :50,000 records, 138 postal codes, 3-year date range (2023–2026).

> *All generated code scripts - Python and HTML - depend strongly on the structure of the provided SQL-lite database snapshot*.
> *If using a distinct data file structure, the generated code needs to be adapted.*  

A [user stories document](https://github.com/kengggg/epihack-arizona/blob/main/analysis/UserStories/EpiHack_UserStories_LLMPrompts.pdf) was created to pair with  LLM prompts for each user case scenarios.

Please see: [**Presentation for the Analysis work**](https://github.com/kengggg/epihack-arizona/blob/main/analysis/UserStories/EpiHack_One_Health_AI.pdf)

Anthropic Claude Cowork using Sonnet 4.6 was prompted with a collection of prompts corresponding to different user cases for users and Public Health Officials. For each user case a Pthon Script was created to extract data for the corresponding dashboard in HTML.


### User Story Prompts — Individual Survey Participants

| Case| User story | Python Script | HTML dashboard | View | Available | 
| :--: | :-- | :--: | :--: | :--: | :-- |
| 1 | Personal Symptom Trajectory | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/symptom_trajectory_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/symptom_trajectory_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/symptom_trajectory_dashboard.png) |  :heavy_check_mark: |
| 2 |  Community Comparison| [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/community_comparison_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/community_comparison_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/community_comparison_dashboard.png) | :heavy_check_mark: |
| 3 |  Emerging Symptom Alert| [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/emerging_signal_alert.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/emerging_signal_alert.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/emerging_signal_alert_dashboard.png) | :heavy_check_mark: |
| 4 | Heat-Related Risk Contextualization | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/heat_risk_contextualization.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/heat_risk_contextualization.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/emerging_signal_alert_dashboard.png) | :heavy_check_mark: |
| 5 | Re-Engagement Motivation | [Python]() | [dashboard]() | [image]() |  |
| 6 | Symptom Co-occurrence Explanation | [Python]() | [dashboard]() | [image]() |  |
| 7 | Vaccination and Prevention Guidance | [Python]() | [dashboard]() | [image]() |  |
| 8 | Animal Exposure Risk Screening | [Python]() | [dashboard]() | [image]() |  |
| 9 | Household Member Risk Assessment | [Python]() | [dashboard]() | [image]() |  |
| 10 | Wastewater Surveillance Contextualization | [Python]() | [dashboard]() | [image]() |  |
| 11 | Loss of Smell or Taste Signal | [Python]() | [dashboard]() | [image]() |  |
| 12 | Demographic Risk Stratification Feedback | [Python]() | [dashboard]() | [image]() |  |
| 13 | Mental Health and Illness Burden | [Python]() | [dashboard]() | [image]() |  |
| 14 |  Occupational Exposure Risk| [Python]() | [dashboard]() | [image]() |  |
| 15 | Rash and Bleeding Symptom Urgency Triage | [Python]() | [dashboard]() | [image]() |  |
| 16 | Progress Feedback and Gamification | [Python]() | [dashboard]() | [image]() |  |
| 17 | Pediatric Household Symptom Reporting | [Python]() | [dashboard]() | [image]() |  |
| 18 | Spanish-Language Health Communication | [Python]() | [dashboard]() | [image]() |  |
| 19 | Tribal Community Health Messaging | [Python]() | [dashboard]() | [image]() |  |
| 20 | Post-Recovery Follow-Up | [Python]() | [dashboard]() | [image]() |  |


### User Story Prompts — Public Health Officials

| Case| User story | Python Script | HTML dashboard | Mock-up | Available | 
| :--: | :-- | :--: | :--: | :--: | :-- |
| 1 | Outbreak Signal Summary Brief | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/situation_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/situation_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/situation_report_dashboard.png) | :heavy_check_mark: |
| 2 | Geographic Cluster Detection | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/spatial_cluster_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/spatial_cluster_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/spatial_cluster_dashboard.png) | :heavy_check_mark: |
| 3 | One Health Zoonotic Risk Assessment | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/one_health_risk_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/one_health_risk_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/one_health_risk_dashboard.png) | :heavy_check_mark: |
| 4 | Health Equity Surveillance Report | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/health_equity_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/health_equity_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/health_equity_dashboard.png) |  :heavy_check_mark: |
| 5 | Wastewater-Clinical-Community Triangulation | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/multi_source_triangulation.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/multi_source_triangulation.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/multi_source_triangulation.png) | :heavy_check_mark: |
| 6 | Public Health Communication Draft | [Python]() | [dashboard]() | [image]() | :heavy_check_mark: |
| 7 | Demographic Shift Detection Alert | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/demographic_surveillance_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/demographic_surveillance_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/demographic_surveillance_report.png) | |
| 8 | Rare Symptom Cluster Investigation | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/rare_event_cooccurrence_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/rare_event_cooccurrence_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/rare_event_cooccurrence_report.png) | :heavy_check_mark: |
| 9 | Reporter Coverage and Bias Assessment | [Python]() | [dashboard]() | [image]() |  |
| 10 | Seasonal Baseline Recalibration | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/seasonal_decomposition_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/seasonal_decomposition_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/seasonal_decomposition_report.png) | :heavy_check_mark: |
| 11 | Vector-Borne Disease Correlation | [Python](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/arboviral_surveillance_report.py) | [dashboard](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/arboviral_surveillance_report.html) | [image](https://github.com/kengggg/epihack-arizona/blob/main/analysis/dashboard/V2/arboviral_surveillance_report.png) | :heavy_check_mark: |
| 12 | Emergency Resource Allocation Recommendation | [Python]() | [dashboard]() | [image]() |  |
| 13 | School and Childcare Illness Monitoring | [Python]() | [dashboard]() | [image]() |  |
| 14 | Agricultural Worker Surveillance | [Python]() | [dashboard]() | [image]() |  |
| 15 | Tribal Nation Health Situation Report | [Python]() | [dashboard]() | [image]() |  |
| 16 | LLM-Assisted Epidemiological Hypothesis Generation | [Python]() | [dashboard]() | [image]() |  |
| 17 | After-Action Report Synthesis | [Python]() | [dashboard]() | [image]() |  |
| 18 | Media and Infodemic Monitoring | [Python]() | [dashboard]() | [image]() |  |
| 19 | Predictive Risk Modeling Brief | [Python]() | [dashboard]() | [image]() |  |
| 20 | Legislative and Policy Intelligence Brief | [Python]() | [dashboard]() | [image]() |  |


***

Created: 05/22/2026 (C. Lizárraga) <br>
Updated: 05/26/2026 (C. Lizárraga)


