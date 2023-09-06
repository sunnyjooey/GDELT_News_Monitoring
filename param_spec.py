# inclusive
START_DATE = '2023-01-02'
# exclusive: download does not include this day 
END_DATE = '2023-01-08' 

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