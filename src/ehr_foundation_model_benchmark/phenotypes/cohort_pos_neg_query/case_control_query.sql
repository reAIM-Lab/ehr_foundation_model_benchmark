WITH control_cohort AS (SELECT *
                        FROM {{database_name}}.results.yl_pheno
WHERE cohort_name = '{{cohort_name}} at risk (0 yrs)')
    , case_cohort AS (
SELECT *
FROM {{database_name}}.results.yl_pheno
WHERE cohort_name = '{{cohort_name}} Cases')

SELECT DISTINCT cas.subject_id, 'case' as subject_cohort
FROM case_cohort AS cas
         INNER JOIN control_cohort AS con
                    ON con.subject_id = cas.subject_id
WHERE cas.cohort_start_date >= con.cohort_start_date
UNION ALL
SELECT DISTINCT con.subject_id, 'control' as subject_cohort
FROM control_cohort AS con
         LEFT JOIN case_cohort AS cas
                   ON con.subject_id = cas.subject_id
WHERE cas.subject_id IS NULL