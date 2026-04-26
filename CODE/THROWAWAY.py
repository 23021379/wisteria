thing = """[REDACTED_BY_SCRIPT]"Asian, Asian British or Asian Welsh ethnic groups only in household_prop_ons","Black, Black British, Black Welsh, Caribbean or African ethnic groups only in household_prop_ons"[REDACTED_BY_SCRIPT]"Flat, maisonette or apartment_prop_ons"[REDACTED_BY_SCRIPT]"""
thing_list=list(thing)
comma_count=0
for i in range(len(thing_list)):
    if thing_list[i] == ',':
        comma_count+= 1
print("There are", comma_count, "[REDACTED_BY_SCRIPT]")