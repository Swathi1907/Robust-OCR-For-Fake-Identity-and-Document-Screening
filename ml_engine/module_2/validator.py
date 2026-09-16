from mrz.checker.td3 import TD3CodeChecker

class invalidPassport(Exception):
    pass

def sanitize_td3_mrz(mrz_text: str) -> str:
    lines = [line.strip().replace(" ", "") for line in mrz_text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return mrz_text

    l1 = lines[0][:44]
    l2 = lines[1][:44]

    # Fix country code (indices 10:13) in line 2 if read as '0' instead of 'O'
    if len(l2) >= 13:
        country_part = l2[10:13].replace('0', 'O')
        l2 = l2[:10] + country_part + l2[13:]

    return f"{l1}\n{l2}"

def validate_mrz(mrz : str) -> dict:
    
    try:
        checker = TD3CodeChecker(sanitize_td3_mrz(mrz))
    except Exception as e:
        return {
            "status" : False
        }
    
    
    
    if bool(checker):
        fields = checker.fields()
        return {
            "status" : True,
            "name" : fields.name,
            "surname" : fields.surname,
            "country" : fields.country,
            "nationality" : fields.nationality
        }
   
    return {
        "status" : False
    }