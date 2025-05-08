import meds_reader
import femr.datasets

path_to_extract= "/user/zj2398/long_context_clues/data/mimic/meds_v0.3_reader"
femr_db = meds_reader.SubjectDatabase(path_to_extract)

count = 0 
for subject_id in femr_db:
    count += 1
    # print(f"{subject_id}th patient")
    for event in femr_db[subject_id].events:
        if event.code == "MIMIC_IV_ITEM/227429":
        # if event.text_value != '' and event.text_value is not None: # `value` is not None
            print(event.time,event.text_value,event.numeric_value,event.unit)
    if count >= 1000:
        break
    # break
        # print(event.value)


# MIMIC_IV_ITEM/225754