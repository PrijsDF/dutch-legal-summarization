import time
from pprint import pprint
import sys

import pandas as pd
from transformers import MBartTokenizer, MBartForConditionalGeneration
# from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

from src.utils import DATA_DIR, MODELS_DIR, load_dataset

# Some pandas options that allow to view all collumns and rows at once
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 400)
pd.options.display.width = None
# This will prevent a warning from happening during the interpunction removal in the LDA function
pd.options.mode.chained_assignment = None

# load pretrained mBERT and a pretrained tokenizer for Semantic Coherence prediction
mbart_model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
mbart_tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')

#t5_model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")
#t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")


def main():
    """View Open Rechtspraak dataset with pandas."""
    # Load the raw dataset
    all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim', use_dask=True)

    first_cases = all_cases.tail()
    case_text = first_cases.iloc[3]['description']

    text = """Een deel van de Russische troepen die zich aan de grens met Oekraïne bevinden, keert terug naar hun bases. Maar grootschalige oefeningen gaan wel door, schrijft het Russische persbureau Interfax op basis van een verklaring van het ministerie van Defensie. Deze aankondiging komt een dag voordat volgens Amerikaanse bronnen een aanval op Oekraïne zou kunnen beginnen en op de dag dat de Duitse bondskanselier Scholz president Poetin bezoekt.

"De eenheden van de zuidelijke en westelijke militaire districten, die hun taken hebben voltooid, beginnen met de verplaatsing naar hun militaire garnizoenen", zegt een woordvoerder van het Russische ministerie van Defensie. Maar grootschalige oefeningen gaan dus door. "Vrijwel alle militaire districten, vloten en luchtlandingstroepen nemen eraan deel", citeert Interfax de woordvoerder.

Hoeveel militairen nu vertrekken, is niet duidelijk. Rusland heeft ongeveer 130.000 troepen dicht bij de grens met Oekraïne opgesteld. Ongeveer 30.000 van hen nemen deel aan militaire oefeningen in Belarus. Die oefeningen lopen af op 20 februari.

    """

    # text = "Meine Freunde sind cool, aber sie essen zu viel Kuchen."

    inputs = mbart_tokenizer(case_text, truncation=True, return_tensors="pt")

    # Generate Summary
    summary_ids = mbart_model.generate(inputs["input_ids"], num_beams=4, max_length=128)

    print(mbart_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))


if __name__ == '__main__':
    main()