## Concept Sets

This folder contains all the concept codes used in all concept sets within each cohort definition.

Each zip file will contain three CSVs:

1. `conceptSetExpression.csv`
All the original codes used within the definition for each concept set. Different concept sets will have a different 'Concept Set ID', as well as a different 'Name'. This file will also show you the logic of 
- isExclude: if this concept is serving as an exclusion to the final concept set expression
- isDescendant: if this concept should pull in descendants from concept_ancestor
- isMapped: if this concept should pull in mapped concepts found by using the concept_relationship’s ‘Maps To’ relationship. In other words, this will include all the source concept codes, like ICD9/10 codes.

> Note: Most of our cohort definitions kept isMapped to be False, as our data is harmonized to OMOP standard concepts. But if your site mainly utilize source concept codes, you can switch mapped to True.

2. `includedConcepts.csv`
This contains all the included concept codes within each concept set, organized by 'Concept Set ID'. In other words, this will show you all the decendent codes utilized in the concept sets.

3. `mappedConcepts.csv`
This contains all the mapped concepts within each concept set, organized by 'Concept Set ID'. In other words, this will show you all the source codes that were mapped to the included standard concepts within each concept set.
