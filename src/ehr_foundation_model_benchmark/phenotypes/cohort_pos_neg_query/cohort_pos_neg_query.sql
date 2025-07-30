--yl_pheno contains all cohort tables, including columns: cohort_definition_id, subject_id, cohort_start_date, cohort_end_date
WITH at_risk_cohort_visit AS (
    SELECT DISTINCT
        yp.subject_id,
        vo.visit_occurrence_id,
        vo.visit_start_datetime,
        yp.cohort_start_date AS at_risk_start_date,
        yp.cohort_end_date AS obs_end_date
    FROM {{database_name}}.results.yl_pheno yp
    JOIN {{database_name}}.dbo.visit_occurrence vo ON vo.person_id = yp.subject_id
    JOIN {{database_name}}.results.phenotype_temp_sample pts ON yp.subject_id = pts.subject_id
    JOIN {{database_name}}.dbo.observation_period op ON yp.subject_id = op.person_id
    WHERE yp.cohort_name = '{{cohort_name}} at risk (0 yrs)'
    AND vo.visit_start_datetime >= yp.cohort_start_date
    AND vo.visit_start_datetime < yp.cohort_end_date
    AND vo.visit_start_datetime >= DATEADD(YEAR, {{min_obs_years}}, op.observation_period_start_date)
),
    --For every visit, there needs to be at least one condition or drug event in the observation window before the visit
    at_risk_with_condition AS (
        SELECT DISTINCT vo.subject_id,
            vo.visit_occurrence_id,
            CASE WHEN count(co.condition_occurrence_id) >= 1 THEN 1
                ELSE 0
                END AS condition_exists
            FROM at_risk_cohort_visit vo
            LEFT JOIN {{database_name}}.dbo.condition_occurrence co ON vo.subject_id = co.person_id
            WHERE co.condition_start_datetime >= DATEADD(YEAR, -{{min_obs_years}}, vo.visit_start_datetime)
            AND co.condition_start_datetime <= vo.visit_start_datetime
            GROUP BY vo.subject_id, vo.visit_occurrence_id

    ),
    at_risk_with_drug AS (
        SELECT DISTINCT vo.subject_id,
            vo.visit_occurrence_id,
            CASE WHEN count(de.drug_exposure_id) >= 1 THEN 1
                ELSE 0
                END AS drug_exists
            FROM at_risk_cohort_visit vo
            LEFT JOIN {{database_name}}.dbo.drug_exposure de ON vo.subject_id = de.person_id
            WHERE de.drug_exposure_start_datetime >= DATEADD(YEAR, -{{min_obs_years}}, vo.visit_start_datetime)
            AND de.drug_exposure_start_datetime <= vo.visit_start_datetime
            GROUP BY vo.subject_id, vo.visit_occurrence_id
    ),

    final_at_risk_cohort AS (
                SELECT DISTINCT
                    vo.subject_id,
                    vo.visit_start_datetime,
                    vo.at_risk_start_date,
                    vo.obs_end_date,
                    con.condition_exists,
                    drg.drug_exists
                FROM at_risk_cohort_visit vo
                LEFT JOIN at_risk_with_condition con
                    ON vo.subject_id = con.subject_id AND vo.visit_occurrence_id = con.visit_occurrence_id
                LEFT JOIN at_risk_with_drug drg
                    ON vo.subject_id = drg.subject_id AND vo.visit_occurrence_id = drg.visit_occurrence_id
                WHERE (con.condition_exists + drg.drug_exists) >= 1
    ),

    case_cohort AS (
    SELECT yp.subject_id as case_subject_id, yp.cohort_start_date AS case_start_date
    FROM {{database_name}}.results.yl_pheno yp
    JOIN {{database_name}}.results.phenotype_temp_sample pts ON yp.subject_id = pts.subject_id
    WHERE cohort_name = '{{cohort_name}} Cases'
)
SELECT DISTINCT subject_id, visit_start_datetime as prediction_time, MAX(boolean_value) as boolean_value
FROM (SELECT *,
             CASE
                 WHEN case_start_date < visit_start_datetime THEN -1
                 WHEN case_start_date <= DATEADD(YEAR, {{prediction_years}}, visit_start_datetime) THEN 1
                 ELSE 0
                 END AS boolean_value
      FROM final_at_risk_cohort
               LEFT JOIN case_cohort ON subject_id = case_subject_id) t
WHERE boolean_value != -1
GROUP BY subject_id, visit_start_datetime