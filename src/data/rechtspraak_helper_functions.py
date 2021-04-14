from bs4 import BeautifulSoup
from datetime import datetime


def parse_xml(file_name):
    # Passing the stored data inside the beautifulsoup parser
    file_xml = BeautifulSoup(file_name, 'xml')

    # Find the RDF info, case summary and case description
    case_rdf = file_xml.find('rdf:RDF')
    case_summary = file_xml.find('inhoudsindicatie')
    case_description = file_xml.find('uitspraak')

    # For some cases the term 'conclusie' is used instead of 'uitspraak'
    if case_description is None:
        case_description = file_xml.find('conclusie')

    return case_rdf, case_summary, case_description


def get_document_attributes(case_rdf):
    rdf_tags = {}

    # Tag pointer dictionary for single tag items
    # Cardinalities are taken from chapter 12 of the pdf:
    # ./references/Rechtspraak data information/Technische-documentatie-Open-Data-van-de-Rechtspraak.pdf
    # Cardinality: 0/1 - 1
    tag_pointer_dict = {
        'identifier': 'dcterms:identifier',  # 1 - 1
        # 'format':'dcterms:format', # 1 - 1
        'seat_location': 'dcterms:spatial',  # 0 - 1
        'publisher': 'dcterms:publisher',  # 1 - 1
        # 'language':'dcterms:language', # 1 - 1
        'creator': 'dcterms:creator',  # 1 - 1
        'case_type': 'dcterms:type'  # 1 - 1
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
        try:
            rdf_tags[tag] = case_rdf.find(pointer).text
        except:
            rdf_tags[tag] = 'none'

    # Lets do the multi tags
    for tag, pointer in multi_tag_pointer_dict.items():
        rdf_tags[tag] = ()

        tag_mentions = case_rdf.find_all(pointer)
        for mention in tag_mentions:
            rdf_tags[tag] += (mention.text,)

    # Date tags (%Y-%m-%d)
    for tag, pointer in date_tag_pointer_dict.items():
        try:
            tag_text = case_rdf.find(pointer).text
            rdf_tags[tag] = datetime.strptime(tag_text, '%Y-%m-%d').date()
        except:
            rdf_tags[tag] = 'none'

    # Datetime tags (%Y-%m-%dT%H:%M:%S)
    for tag, pointer in datetime_tag_pointer_dict.items():
        try:
            tag_text = case_rdf.find(pointer).text
            rdf_tags[tag] = datetime.strptime(tag_text, '%Y-%m-%dT%H:%M:%S')
        except:
            rdf_tags[tag] = 'none'

    return rdf_tags
