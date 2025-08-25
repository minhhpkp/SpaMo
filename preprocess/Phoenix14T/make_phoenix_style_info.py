# make_phoenix_style_info.py
import numpy as np

data = {
    # 0: {
    #   'fileid': '25October_2010_Monday_tagesschau-17',
    #     'folder': 'test/25October_2010_Monday_tagesschau-17/*.png', 
    #     'signer': 'Signer01', 
    #     'gloss': 'REGEN SCHNEE REGION VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN SEHEN', 
    #     'text': 'regen und schnee lassen an den alpen in der nacht nach im norden und nordosten fallen hier und da schauer sonst ist das klar',
    #       'num_frames': 181, 
    #       'original_info': '25October_2010_Monday_tagesschau-17|25October_2010_Monday_tagesschau-17/1/*.png|-1|-1|Signer01|REGEN SCHNEE REGION VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN SEHEN|regen und schnee lassen an den alpen in der nacht nach im norden und nordosten fallen hier und da schauer sonst ist das klar',
    #         'tag': 'phoenix14t', 
    #         'en_text': 'Rain and snow let go of the Alps on the night in the north and northeast fall here and there. Otherwise it is clear.',
    #           'es_text': 'La lluvia y la nieve dejan ir a los Alpes en la noche en el norte y el noreste caen aquí y allá.',
    #             'fr_text': 'La pluie et la neige ont lâché les Alpes de la nuit au nord et au nord-est ici et là.'
    #   },
    0: {'fileid': '25October_2010_Monday_tagesschau-24', 'folder': 'test/25October_2010_Monday_tagesschau-24/*.png', 'signer': 'Signer01', 'gloss': 'DONNERSTAG NORDWEST REGEN REGION SONNE WOLKE WECHSELHAFT DANN FREITAG AEHNLICH WETTER', 'text': 'am donnerstag regen in der nordhälfte in der südhälfte mal sonne mal wolken ähnliches wetter dann auch am freitag', 'num_frames': 150, 'original_info': '25October_2010_Monday_tagesschau-24|25October_2010_Monday_tagesschau-24/1/*.png|-1|-1|Signer01|DONNERSTAG NORDWEST REGEN REGION SONNE WOLKE WECHSELHAFT DANN FREITAG AEHNLICH WETTER|am donnerstag regen in der nordhälfte in der südhälfte mal sonne mal wolken ähnliches wetter dann auch am freitag', 'tag': 'phoenix14t', 'en_text': 'On Thursday in the northern half in the southern half of the sun, clouds of similar weather also stimulate on Friday.', 'es_text': 'El jueves en la mitad norte en la mitad sur del sol, las nubes de clima similar también se estimulan el viernes.', 'fr_text': 'Jeudi dans la moitié nord dans la moitié sud du soleil, des nuages \u200b\u200bde temps similaire stimulent également vendredi.'},
    1: {}  # dummy so len(data) == 2 -> num = 1 -> one iteration
}

np.save('test_info.npy', data, allow_pickle=True)
np.save('test_info_ml.npy', data, allow_pickle=True)
print("Saved test_info.npy")
