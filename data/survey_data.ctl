OPTIONS (SKIP=1)
LOAD DATA
INFILE 'MOCK_DATA.csv'
INTO TABLE survey_data
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
  id,
  occupation,
  state,
  district,
  gender,
  income,
  year
)
