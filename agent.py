"""
'Training':
For each factor, there will be a report on how this one factor influences price, and an analysis of the dataset that provide's the quantitative affect it has
 - i.e how a percentage change in factor affects price
On top of that,there will also be a report on how the factors interact with each other, and how they affect price
 - i.e. how a percentage change in factor A and B together affects price

prediction:
For each factor, the report and analysis of the factor as well as the interaction with other factors will be used to predict the price of the house
 The information will be given to several instances of gemini, each with differing temperatures. 
 Each instance will be given the following options:
  Requests:
    the instance can request the output of another instance - this could be an instance of another factor, or a different temperature of the same factor
    the istance can also request supplementation - i.e. it can request to use the grounding tool to get more information on the factor and what questions to ask
    the instance can also request other instances inputs/datasets
 
 The initial instances outputs will look like:
  temperature, factor, factor interactions, predictions, requests made, and justifications for predictions (citing the information used)
  
 After each 1st instance has generated its output, the 2nd layer is tasked with using the dataset and the 1st layer'[REDACTED_BY_SCRIPT]'s
  justifications and predictions to help it form its own prediction.
    This instance will also be able to request information from the 1st layer instances, and will be able to request information from the dataset.
     I.e. if the output wasn't detailed enough, or was confusing, it can request more information from the 1st layer instance.
  
"""