import datetime as dt
import pandas as pd

COUNTRY_CODES = 'MI'
CO = 'malawi'

### CONSTANT

DATABASE_NAME = f'{CO}_news'

EVENT_TABLE = 'brz_gdelt_events_raw'

EMBED_TABLE = 'brz_gdelt_emb_title_raw'

ARTICLE_TEXT_TABLE = 'brz_article_text_scrape'

CLEAN_TABLE = 'slv_event_title_text_clean'

SUMMARY_TABLE = 'gld_openai_summary'

VIZ_TABLE = 'viz_main'

ERROR_TABLE = 'error_table'

ADMIN_TABLE = 'brz_admin_combine'

SHAPEFILE = '/dbfs/FileStore/df/shapefiles/malawi_admin1'

CAMEO_TABLE =  pd.read_csv('/dbfs/user/hive/warehouse/malawi_news.db/cameo.csv',
                           header=None,
                           names=['code', 'Events', 'additional_info_1', 'additional_info_2'],
                           delimiter=',',
                           dtype={'code': 'str'})
                    
TARGET_CAMEO = ['14','15','17','18','19']