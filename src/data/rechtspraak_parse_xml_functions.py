from bs4 import BeautifulSoup
from datetime import datetime


def parse_xml(file_name):
    """ Using bs4, the case document is parsed. Three parts are extracted: the rdf; containing mostly meta-information,
    the case summary, and the case description. Returns a dictionary of all relevant pieces of information of the case.
    """
    # Passing the stored data inside the beautifulsoup parser
    file_xml = BeautifulSoup(file_name, 'xml')

    # Find the RDF info, case summary and case description
    case_rdf = file_xml.find('rdf:RDF')
    case_summary = file_xml.find('inhoudsindicatie')
    case_description = file_xml.find('uitspraak')

    # For some cases the element 'conclusie' is used instead of 'uitspraak'
    if case_description is None:
        case_description = file_xml.find('conclusie')

    # Parse case_rdf to get document meta attributes
    case_content = get_document_attributes(case_rdf)

    # Will tell us whether summary, description, or both are missing
    missing = tuple()

    # Extract the summary text
    if case_summary is not None:
        case_content['summary'] = case_summary.get_text('|', strip=True)
    else:
        case_content['summary'] = 'none'
        missing = missing + ('summary',)

    # Extract the description text
    if case_description is not None:
        case_content['description'] = case_description.get_text('|', strip=True)
    else:
        case_content['description'] = 'none'
        missing = missing + ('description',)

    # Tuples can't be stored in a parquet
    missing = 'none' if len(missing) == 0 else '|'.join(missing)

    # Add information about missing parts to meta df
    case_content['missing_parts'] = missing

    return case_content


def get_document_attributes(case_rdf):
    """ Parses case information that is contained in the rdf tag of the XML file, returning a dictionary containing
    the relevant pieces of (meta) information of a case.
    """
    rdf_tags = {}

    # Tag pointer dictionary for single tag items
    # Cardinalities are taken from chapter 12 of the pdf:
    # ./references/Rechtspraak data information/Technische-documentatie-Open-Data-van-de-Rechtspraak.pdf
    # Cardinality: 0/1 - 1
    tag_pointer_dict = {
        'identifier': 'dcterms:identifier',  # 1 - 1
        'seat_location': 'dcterms:spatial',  # 0 - 1
        # 'publisher': 'dcterms:publisher',  # 1 - 1, omit because always equal to 'Raad voor de Rechtspraak'
        'creator': 'dcterms:creator',  # 1 - 1
        'case_type': 'dcterms:type',  # 1 - 1
        # 'language':'dcterms:language', # 1 - 1, omit because always equal to NL
        # 'format':'dcterms:format', # 1 - 1, omit because always equal to text/xml
    }

    # Tag pointer dictionary for multi tag items
    # Cardinality 0/1 - many
    multi_tag_pointer_dict = {
        'jurisdiction': 'dcterms:subject',  # 0 - many
        'case_number': 'psi:zaaknummer',  # 1 - many
        'procedures': 'psi:procedure',  # 0 - many
        'references': 'dcterms:references',  # 0 - many
        'relation': 'dcterms:relation'  # 0 - many
    }

    # Date type cases
    date_tag_pointer_dict = {
        'issue_date': 'dcterms:issued',  # 1 - 1
        'judgment_date': 'dcterms:date'  # 1 - 1
    }

    # Datetime type cases
    datetime_tag_pointer_dict = {
        'modified': 'dcterms:modified'  # 1 - 1
    }

    # Start with the single tag items
    for tag, pointer in tag_pointer_dict.items():
        found_tag = case_rdf.find(pointer)
        if found_tag is not None:
            rdf_tags[tag] = str(found_tag.text)
        else:
            rdf_tags[tag] = 'none'

    # Lets do the multi tags
    for tag, pointer in multi_tag_pointer_dict.items():
        found_tags = ()
        tag_mentions = case_rdf.find_all(pointer)
        for mention in tag_mentions:
            found_tags += (mention.text,)

        rdf_tags[tag] = 'none' if len(found_tags) == 0 else '|'.join(found_tags)

    # Date tags (%Y-%m-%d)
    for tag, pointer in date_tag_pointer_dict.items():
        found_tag = case_rdf.find(pointer)
        if found_tag is not None:
            rdf_tags[tag] = datetime.strptime(found_tag.text, '%Y-%m-%d')  # .date()
        else:
            rdf_tags[tag] = 'none'

    # Datetime tags (%Y-%m-%dT%H:%M:%S)
    for tag, pointer in datetime_tag_pointer_dict.items():
        found_tag = case_rdf.find(pointer)
        if found_tag is not None:
            rdf_tags[tag] = datetime.strptime(found_tag.text, '%Y-%m-%dT%H:%M:%S')
        else:
            rdf_tags[tag] = 'none'

    return rdf_tags
