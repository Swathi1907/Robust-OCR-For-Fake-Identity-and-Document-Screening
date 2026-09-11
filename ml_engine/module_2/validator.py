from mrz.checker.td3 import TD3CodeChecker

class invalidPassport(Exception):
    pass

def validate_mrz(mrz : str) -> dict:
    
    try:
        checker = TD3CodeChecker(mrz)
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