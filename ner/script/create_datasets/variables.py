DETERMINANTS = [
    ["l'", "d'", "L'", "D'"],
    ["la ", "La ", "le ", "Le "], 
    ["les", "Les ", "des ", "Des "]
]

# Shareholdership : Il y a eu quelque convertion entre ce label et moneyamount + currency puisque c'etait davantage des montants (principalement dans A/Societe_generale)
# Financing : Meme chose que Shareholdership
# Location : Tous convertie en WorldRegion, Country, LocalRegion ou City
# GeopoliticalEntity: Transformer ceux qui sont des locations (donc WorldRegion, Country, LocalRegion ou City) et des organizations et supprimer les autres
# Activity
LABELS_TO_SKIP = set([
    'Investment', # {'investissements industriels', 'payer'}
    'Transaction', # {'dividende', 'cession'}
    'Penalty', # {'sanction', 'pénalité financière'}
    'Demerger', # {'directeur production', 'scission'}
    'Fine', # {'amende'}
    'Sanction', # {'procès', 'sanctions économiques'}
    'IPO', # {'entrer en bourse', 'IPO', 'cotée', "l'introduction en Bourse", 'cotée en bourse', 'cotation', 'cotés à wall street', 'être coté', 'cotées', 'se faire coter', 'cotés', 'se coter'}
    'Merger', # {'fusions', 'fusions-acquisitions', 'fusion', 'se rapprocher', 'M&A', 'rallier', 'mégafusion', 'rachat'}
    'BusinessDeal', # {'transactions journalières sur les monnaies', 'vendre', 'collaborent', "l'achat", 'accord', 'échanges commerciaux', 'loue', 'versée', 'revendues', 'achats', 'commerce', 'cessions financières', 'céder le contrôle', 'cliente', 'contrat de vente', 'cession', 'OPA', 'commande', 'accord de service', 'signent des licences', 'versement'}
    'Acquisition', # {"l'acquisition", "L'acquisition", "option d'achat", 'cession', 'achat', "s'approprier", "l'OPA", 'acquisitions', 'reprendre', 'racheter', "l'OPAS", "L'OPA", 'avait été reprise', 'acheté', 'reprennent', 'rachetant', "s'est offert", 'acquis', 'acquérir', 'fusions-acquisitions', 'rachète', 'opa', 'rachat', 'racheté', "l'offre publique d'achat", 'acquisition', 'a acquis la part', 'reprise', 'offre hostile'}
    'Document', # {'loi Hamon', 'Paying Taxes', 'la loi Rebsamen', 'communiqué', 'diplômes', 'contrats', 'rapport vickers', 'contrat à durée indéterminée', 'MBA', 'enquête', 'la loi du 6 août 2015', 'contrat', 'accord de joint-venture', 'DEA', 'master ingénierie financière', 'contrat initial', 'loi El Khomri', 'accord nucléaire', 'la loi Consommation', 'Acemo Dialogue social en entreprise', 'diplômé', 'accords de collaboration et de licences croisées', 'master', 'la loi Hamon', 'la loi Consommation,', "contrats d'intérim", 'la loi n° 2013-504'}
    'Shareholdership', # {'actions', "d'actionnaires", 'entre au capital', "s'invite au capital", 'actionnariats', "l'actionnariat", 'titres', 'Actionnariat', 'détiennent', 'participation', '30% du capital', "l'action", 'posséder', 'actionnariat', 'actionnaires', 'actionnaire'}
    'Financing', # {'investi', 'lever', 'investis', 'investissant', 'La levée de fonds', 'réinvesti', 'investissement', 'consacre', 'investirait', 'prêté', 'les investissements du fonds', 'financer', 'levée de fonds', 'investissements industriels', 'financements', 'refinancement', 'autofinancement', 'investissements de r&d ou de capacités de production', 'investir', 'se financer', 'investissements', 'placement privé', 'autofinancement financière', 'levant', 'financé', 'investissements spéculatifs', 'FINANCEMENT', "s'autofinancer", 'finance', 'investissements financiers', "l'autofinancement", 'réinvestir', 'investit', 'financement', 'investissements de portefeuille'}
    'TangibleAsset',
    'FinancialAsset',
    'Asset', # {'actions', 'titre', 'barils', etc.}
    'GeopoliticalEntity', # {'les gouvernants', "ministère de l'Intérieur", 'Economie et des Finances', 'Ministère de la Culture et de la Communication', 'eurozone', "ministère de l'Industrie, de la Poste et des Télécommunications", 'etats', "Ministère de l'Agriculture, de l'Agroalimentaire et de la Forêt", 'tribunal de commerce', 'ministère des Entreprises et du Développement économique', "l'Etat", "ministères de l'Industrie et de la Recherche", "l'administration des douanes", 'la Cour des comptes', 'ministère délégué chargé de la Santé et de la Protection sociale', "ministère de l'Industrie", 'fmi', "ministère de l'Emploi et de la Solidarité", "ministère de l'Economie et des Finances", "Ministère du Logement et de l'Egalité des territoires", 'zone euro', 'bric', 'etat', 'g20', etc.}
    'Activity', # {'services financiers', 'la vente de produits de santé et de beauté', 'location de vidéo', 'high-tech', "l'activité Travaux", 'technologies mobiles', 'Déchets spéciaux', 'constructeur automobile', 'avionneur', 'ferroviaire', 'propulsion aéronautique et spatiale', 'les voyages daffaires et de loisirs', "l'aéronautique", 'conseils investisseurs :due diligence comptable', 'technologiques', 'la fibre optique', 'cosmétiques', 'le capot', 'livraison express', "sociétés de gestions d'actifs", "l'audiovisuel", "aviation d'affaires", 'télécommunications', "éditeur de logiciels d'assurance", "l'industrie", 'ingénierie', "fonds d'investissement ou de capital investissement", 'équipements de sécurité pour la maison', 'transformateur de matières plastiques', 'sous-traitant aéronautique', 'courtage en formation', 'la restauration collective', 'équipementiers automobiles', "l'automobile", 'la restauration', 'le coaching', 'travaux de construction', etc. }
    'Role'
])
LABELS_TO_CHANGE = {
    'Agent': 'Organization',
    'Association': 'Organization',
    'Media': 'Organization',
    'Company': 'Organization',
    'WorldRegion': 'Location',
    'Country': 'Location',
    'LocalRegion': 'Location',
    'City': 'Location'
}