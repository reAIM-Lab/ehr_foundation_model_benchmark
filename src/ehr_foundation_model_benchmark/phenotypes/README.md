## Phenotypes

We curated 11 phenotypes as downstream tasks. Each phenotype contains two json files. One for defining the case cohort, another for defining the at risk cohort.

To use the json files, there are three approaches depending on if a local OHDSI Atlas instance is available.

1. If local OHDSI Atlas instance exist, you can load the phenotype json files into your cohort definition section and generate the cohorts directly.
2. If local OHDSI Atlas instance doesn't exist, you can use this [CohortGenerator R package](https://github.com/OHDSI/CohortGenerator?tab=readme-ov-file) to execute the json files and generate cohort.
3. Use the [OHDSI community Atlas instance](https://atlas-demo.ohdsi.org/#/home) to read the cohort definitions and concept sets: "Cohort Definitions" through "New Cohort" -> "Export" -> "JSON" -> Copy the phenotype json files into the box -> "Reload". Then adapt the logic and concept codes to the format that suits your data. We have also uploaded all the used concept sets with their included and mapped concept codes in [cohort_concept_sets](./cohort_concept_sets).