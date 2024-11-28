# 스트림릿 라이브러리를 사용하기 위한 임포트
import streamlit as st
import pandas as pd
import numpy as np
import cx_Oracle as co
from datetime import datetime, timedelta
import altair as alt

color_map = ['#F58220', '#043B72','#00A9CE', '#F0B26B', '#8DC8E8','#CB6015','#AE634E', '#84888B','#7EA0C3', '#C2AC97', '#0086B8']

st.set_page_config(page_title='Invest Pool Dashboard', layout='wide')
today = datetime.today().strftime('%y-%m-%d')

def connect_to_db(username, password, host, port, service_name):
    dsn = co.makedsn(host, port, sid=service_name)
    conn = co.connect(username, password, dsn)
    return conn

db_secrets = st.secrets["m_db"]
username = db_secrets["username"]
password = db_secrets["password"]
host = db_secrets["host"]
port = db_secrets["port"]
service_name = db_secrets["service_name"]

@st.cache_data
def db_connect(sql, username=username, password=password, host=host, port=port, service_name=service_name):
    conn = connect_to_db(username, password, host, port, service_name)
    df = pd.read_sql(sql, con=conn)
    conn.close()
    return df

def calculate_period_return(group):
    start_value = group.iloc[0]["CLOSE_INDEX"]
    end_value = group.iloc[-1]["CLOSE_INDEX"]
    return np.round((end_value - start_value)/ start_value * 100,2)  # 수익률 (%)

def get_period_value(group):
    start_value = group.iloc[0]["CLOSE_INDEX"]
    end_value = group.iloc[-1]["CLOSE_INDEX"]
    return start_value, end_value

def convert_name(data:list):
    return [x.replace('투자풀', '').replace('[주식]','').replace('(주식)','') for x in data]

sql_date = '''
		SELECT 
	                BF_TRD_DT
	                ,BF2_TRD_DT
            	FROM AMAKT.FSBD_DT_INFO 
            	WHERE BASE_DT = '{targetdate}'
	'''.format(targetdate=today)

def main() :
    # 사이드바(Sidebar)
    date_info = db_connect(sql_date)
    default_date = date_info['BF_TRD_DT'][0]
    default_date_before = date_info['BF2_TRD_DT'][0]
    
    with st.sidebar:
        st.title('Invest Pool Dashboard')
        st.markdown(
                        """
                        <p style="font-size:14px; color:#7EA0C3;">
                            <em>Prototype v0.1.3</em>
                        </p>
                        """,
                        unsafe_allow_html=True
                    )

        startdate = st.date_input('Start Date: ', value=default_date_before)
        enddate = st.date_input('End Date: ', value=default_date)

    mkt_index_sql = '''
                WITH BM_INFO AS (	
                    SELECT 
                            A.WKDATE, A.IDX_CD AS INDEX_ID
                            , CASE WHEN B.IDX_NM = 'KOSPI 200 ESG INDEX' THEN '코스피 200 ESG' 
                                WHEN B.IDX_NM = '코리아밸류업지수(Price, KRW, D)' THEN '코리아밸류업지수'					
                            END AS KFNAME
                            , A.IDX_VAL AS CLOSE_INDEX
                    FROM 
                        AMAKT.E_MA_IDX_DATA A
                    LEFT JOIN 
                        AMAKT.E_MA_IDX_MAST B
                        ON A.IDX_CD = B.IDX_CD
                    WHERE B.IDX_CD IN ('120017', '1X2019')

                    UNION ALL

                    SELECT
                        WKDATE
                        , INDEX_ID
                        , KFNAME
                        , CLOSE_INDEX
                    FROM AMAKT.E_MA_FN_CLOSE_INDEX
                    WHERE KFNAME IN ('코스피 200', '코스피','코스닥')
                        )
                , RAWDATA AS (
                            SELECT 
                                A.WKDATE
                                , A.KFNAME
                                , A.CLOSE_INDEX
                            FROM BM_INFO A
                            LEFT JOIN AMAKT.FSBD_DT_INFO DT
                                ON A.WKDATE = DT.BASE_DT
                            WHERE DT.TRD_DT_YN = 'Y'
                            )

		            SELECT *
                    FROM RAWDATA 
                    WHERE WKDATE BETWEEN '{startdate}' AND '{enddate}'
    '''.format(startdate=startdate, enddate=enddate)
    
    bm_sql = '''
            SELECT B.WKDATE AS 일자
                , B.JNAME AS 종목명
                , B.INDUSTRY_LEV1_NM AS 대분류
                , B.INDUSTRY_LEV2_NM AS 중분류
                , B.INDUSTRY_LEV3_NM AS 소분류
                , A.INDEX_NAME_KR AS BM명 
                , A.INDEX_WEIGHT AS BM비중
                , ROUND(B.RATE/100, 4) AS 일수익률

            FROM AMAKT.E_MA_KRX_PKG_CONST A

            LEFT JOIN AMAKT.E_MA_FN_JONGMOK_INFO B
                ON A.FILE_DATE = B.WKDATE
                AND A.CONSTITUENT_ISIN = B.STDJONG
                AND B.INDEX_ID IN ('I.001', 'I.201')
            WHERE A.FILE_DATE BETWEEN '{startdate}' AND '{enddate}'
            AND A.INDEX_CODE1 IN ('1', '2')  -- 1:코스피, 2:코스닥
            AND A.INDEX_CODE2 IN ('001')  -- 001:코스피, 029:코스피200
            ORDER BY INDEX_NAME_KR DESC, INDEX_WEIGHT DESC
        '''.format(startdate=startdate, enddate=enddate)

    bm_df = db_connect(bm_sql)
    bm_df['일자'] = pd.to_datetime(bm_df['일자']).dt.date
    
    bm_df_lastday = bm_df[bm_df['일자'] == bm_df['일자'].max()].drop('일자', axis=1).set_index('종목명')


    df_sql = '''
            WITH BM_WGT AS (
				SELECT *
				FROM (
						SELECT B.WKDATE, B.JNAME
							, B.STDJONG
							, B.INDUSTRY_LEV1_NM
							, B.INDUSTRY_LEV2_NM
							, B.INDUSTRY_LEV3_NM
							, A.INDEX_NAME_KR
							, A.INDEX_WEIGHT
							, ROUND(B.RATE/100, 4) AS "일수익률"
						
						FROM AMAKT.E_MA_KRX_PKG_CONST A
						
						LEFT JOIN AMAKT.E_MA_FN_JONGMOK_INFO B
							ON A.FILE_DATE = B.WKDATE
							AND A.CONSTITUENT_ISIN = B.STDJONG
							AND B.INDEX_ID IN ('I.001', 'I.201')
						WHERE A.FILE_DATE BETWEEN '{startdate}' AND '{enddate}'
						AND A.INDEX_CODE1 IN ('1', '2')  -- 1:코스피, 2:코스닥
						AND A.INDEX_CODE2 IN ('001')  -- 001:코스피, 029:코스피200
						)
				)
				
		, RAWDATA AS (
			
			SELECT     A.BASE_DT
			        , A.FUND_NM
			        , A.KOR_NM
			        , A.SEC_WGT
			         FROM       
			                    (  
			                        SELECT   a.BASE_DT
			                                , a.PTF_CD
			                                , a.FUND_NM
			                                , a.ISIN_CD
			                                , a.KOR_NM
			                                , a.FUND_NAV
			                                , a.FUND_WGT*100 AS FUND_WGT
			                                , a.SEC_WGT*100 AS SEC_WGT
			                        FROM    (       
			                                SELECT  BASE_DT, PTF_CD, FUND_NM, ISIN_CD , KOR_NM, FUND_NAV, ROUND(FUND_WGT, 8) AS FUND_WGT
			                                        , ROUND(EVL_AMT/FUND_NAV,8) AS SEC_WGT
			                                FROM    (                 
			                                            SELECT  e.BASE_DT,k.PTF_CD, k.SUB_PTF_CD , m.KOR_NM AS FUND_NM
			                                                    , NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) AS ISIN_CD
			                                                    , NVL(sm.KOR_NM, em.KOR_NM) AS KOR_NM                             -- 선물 내 이름 사용 후 없으면 주식 이름 사용
			                                                    , n.NET_AST_TOT_AMT AS FUND_NAV , k.WGT AS FUND_WGT
			                                                    , SUM(e.EVL_AMT * NVL(w.WGT, 1)) AS EVL_AMT                        -- 선물 내 비중이 있으면 곱해주고 그게 아니라면 주식이니 그냥 1 곱합
			                                            FROM    AIVSTP.FSBD_PTF_MSTR m
			                                                    INNER JOIN      (       
			                                                                        SELECT   BASE_DT, PTF_CD, ISIN_CD AS SUB_PTF_CD , EVL_AMT , WGT 
			                                                                        FROM     AIVSTP.FSCD_PTF_ENTY_EVL_COMP e1
			                                                                        WHERE        BASE_DT BETWEEN '{startdate}' AND '{enddate}'
			                                                                            AND      DECMP_TCD = 'U'
			                                                                            AND      AST_CD IN ('SFD', 'FND')
			                                                                            AND      PTF_CD = '308611'
			                                                                            AND		 ISIN_CD IN ('KRZ502465730','KRZ502211090','KRZ502465070'
																												,'KRZ502503770','KRZ502511340','KRZ502514660'
																												,'KRZ502515020','KRZ502589850','KRZ502593290')
																					UNION ALL
																					
																					SELECT   BASE_DT, PTF_CD, ISIN_CD AS SUB_PTF_CD , EVL_AMT , WGT 
			                                                                        FROM     AIVSTP.FSCD_PTF_ENTY_EVL_COMP e1
			                                                                        WHERE        BASE_DT BETWEEN '{startdate}' AND '{enddate}'
			                                                                            AND      DECMP_TCD = 'U'
			                                                                            AND      AST_CD IN ('SFD', 'FND')
			                                                                            AND      PTF_CD = '308614'
			                                                                            AND		 ISIN_CD IN ('KRZ502564900','KRZ502564860')
			                                                                      ) k
			                                                               ON      m.PTF_CD = k.SUB_PTF_CD
			                                                               AND     m.KOR_NM LIKE '%주식%'
			                                                     LEFT JOIN          AIVSTP.FSCD_PTF_ENTY_EVL_COMP e
			                                                                ON      k.SUB_PTF_CD = e.PTF_CD 
			                                                                AND     e.BASE_DT = k.BASE_DT
			                                                                AND     e.DECMP_TCD =  'E' --  개별펀드는 'E'로 분해
			                                                                AND     e.AST_CD  IN  ('STK')                     -- 주식과 선물 포함, 참고로 'A'로 분해시 ETF까지는 분해되어 있음         
			                                                    LEFT OUTER JOIN     AMAKT.FSBD_ERM_STK_MAP_MT em                     -- 삼성전자우 와 같은 종목을 삼성전자로 인식하기 위해 필요
			                                                                    ON  e.ISIN_CD = em.ISIN_CD
			                                                    LEFT OUTER JOIN     AMAKT.FSBD_ERM_IDX_ENTY_WGT w                    -- 선물 분해 로직
			                                                                    ON  NVL(em.RM_ISIN_CD, e.ISIN_CD) = w.IDX_CD         -- 선물의 RM_ISIN_CD 를 활용해서 연결
			                                                                    AND e.BASE_DT = w.BASE_DT 
			                                                    LEFT OUTER JOIN     AMAKT.FSBD_ERM_STK_MAP_MT sm                     -- em 테이블에서 했던 것 동일 반복
			                                                                    ON  w.ISIN_CD = sm.ISIN_CD 
			                                                    LEFT OUTER JOIN     AIVSTP.FSCD_PTF_EVL_COMP n 
			                                                                    ON  e.BASE_DT = n.BASE_DT   
			                                                                    AND m.PTF_CD = n.PTF_CD   
			                                            GROUP BY  e.BASE_DT, k.PTF_CD, k.SUB_PTF_CD,  m.KOR_NM,  NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) ,  NVL(sm.KOR_NM, em.KOR_NM) , n.NET_AST_TOT_AMT, k.WGT         
			                                        )
			                                ) a
			                                LEFT OUTER JOIN AMAKT.FSBD_ERM_STK_MAP_MT s
			                                        ON  a.ISIN_CD = s.ISIN_CD
			                             ) A
			
			        INNER JOIN           POLSEL.V_FUND_CD  V
			                        ON   A.PTF_CD = V.AM_FUND_CD    
			                        WHERE FUND_WGT NOT LIKE '100'
			)


            SELECT A.BASE_DT AS 일자, A.FUND_NM AS 펀드명, A.KOR_NM AS 종목명, B.INDUSTRY_LEV1_NM AS 대분류, B.INDUSTRY_LEV2_NM AS 중분류, 
                    B.INDUSTRY_LEV3_NM AS 소분류, B.INDEX_NAME_KR AS BM명, A.SEC_WGT AS 보유비중, B.INDEX_WEIGHT AS BM비중, B.일수익률
            FROM RAWDATA A
            LEFT JOIN BM_WGT B
                ON A.BASE_DT = B.WKDATE
                AND A.KOR_NM = B.JNAME
            WHERE A.KOR_NM IS NOT NULL
            AND B.INDEX_WEIGHT <> 0 
        '''.format(startdate=startdate , enddate= enddate)


    fund_sql = '''
                WITH RAWDATA AS (
                    SELECT  e.BASE_DT,k.PTF_CD, k.SUB_PTF_CD , m.KOR_NM AS FUND_NM
                            , NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) AS ISIN_CD
                            , NVL(sm.KOR_NM, em.KOR_NM) AS KOR_NM                             -- 선물 내 이름 사용 후 없으면 주식 이름 사용
                            , n.NET_AST_TOT_AMT AS FUND_NAV , k.WGT AS FUND_WGT
                            , SUM(e.EVL_AMT * NVL(w.WGT, 1)) AS EVL_AMT                        -- 선물 내 비중이 있으면 곱해주고 그게 아니라면 주식이니 그냥 1 곱합
                    FROM    AIVSTP.FSBD_PTF_MSTR m
                            INNER JOIN		(		
                                                SELECT 	 BASE_DT, PTF_CD, ISIN_CD AS SUB_PTF_CD , EVL_AMT , WGT 
                                                FROM	 AIVSTP.FSCD_PTF_ENTY_EVL_COMP e1
                                                WHERE	     BASE_DT = '{enddate}'
                                                    AND 	 DECMP_TCD = 'U'
                                                    AND      AST_CD IN ('SFD', 'FND')
                                                    AND 	 PTF_CD IN ('308645', '308648','308604','308609','308611','308614','308615','308620', '308626','308627','308644','308650','308671','308673','308683','308691','308703')
                                            ) k
                                    ON      m.PTF_CD = k.SUB_PTF_CD
                                    AND     m.KOR_NM LIKE '%주식%'

                            LEFT JOIN 			AIVSTP.FSCD_PTF_ENTY_EVL_COMP e
                                        ON 		k.SUB_PTF_CD = e.PTF_CD 
                                        AND 	e.BASE_DT = k.BASE_DT
                                        AND 	e.DECMP_TCD =  'E' --  개별펀드는 'E'로 분해
                                        AND 	e.AST_CD  IN  ('STK', 'SFT')  					 -- 주식과 선물 포함, 참고로 'A'로 분해시 ETF까지는 분해되어 있음	
                            LEFT OUTER JOIN 	AMAKT.FSBD_ERM_STK_MAP_MT em                 	 -- 삼성전자우 와 같은 종목을 삼성전자로 인식하기 위해 필요
                                            ON 	e.ISIN_CD = em.ISIN_CD
                            LEFT OUTER JOIN 	AMAKT.FSBD_ERM_IDX_ENTY_WGT w                    -- 선물 분해 로직
                                            ON 	NVL(em.RM_ISIN_CD, e.ISIN_CD) = w.IDX_CD      	 -- 선물의 RM_ISIN_CD 를 활용해서 연결
                                            AND e.BASE_DT = w.BASE_DT 
                            LEFT OUTER JOIN     AMAKT.FSBD_ERM_STK_MAP_MT sm                  	 -- em 테이블에서 했던 것 동일 반복
                                            ON  w.ISIN_CD = sm.ISIN_CD 
                            LEFT OUTER JOIN		AIVSTP.FSCD_PTF_EVL_COMP n 
                                            ON  e.BASE_DT = n.BASE_DT   
                                            AND m.PTF_CD = n.PTF_CD  	       
                                            
                    GROUP BY  e.BASE_DT, k.PTF_CD, k.SUB_PTF_CD,  m.KOR_NM,  NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) 
                            , NVL(sm.KOR_NM, em.KOR_NM) , n.NET_AST_TOT_AMT, k.WGT 	    
                    )
            , LIST_EQ_BM AS
                        (
                        SELECT DISTINCT BASE_DT, SUB_PTF_CD, SUB_F_ID, ZR_CODE, SUB_FUND_NM, SUB_FUND_TYP, SUB_BM_NM
                        FROM POLSEL.V_FUND_EQUITY_BM
                        WHERE BASE_DT = '{enddate}'
                        
                        ) 
                        
            , RAWDATA2 AS (
                            SELECT A.BASE_DT, A.PTF_CD, A.SUB_PTF_CD, A.FUND_NM, ROUND(A.FUND_WGT, 6) AS FUND_WGT
                            FROM RAWDATA A
                            WHERE A.ISIN_CD IS NOT NULL
                            GROUP BY A.BASE_DT, A.PTF_CD, A.SUB_PTF_CD, A.FUND_NM, A.FUND_WGT
                            )

            SELECT PTF_CD, 액티브주식형, 인덱스주식형
            FROM (
                    SELECT DISTINCT PTF_CD, BASE_DT, SUB_FUND_TYP, SUM(FUND_WGT) OVER (PARTITION BY PTF_CD, SUB_FUND_TYP) AS FUND_WGT_SUM
                    FROM (
                            SELECT A.BASE_DT, A.PTF_CD, L.SUB_FUND_NM, L.SUB_FUND_TYP, A.FUND_WGT
                            FROM RAWDATA2 A
                            LEFT JOIN LIST_EQ_BM L
                                ON A.SUB_PTF_CD = L.SUB_PTF_CD
                                AND A.BASE_DT = L.BASE_DT
                        )
                    GROUP BY BASE_DT, PTF_CD, SUB_FUND_TYP, FUND_WGT
                )
            PIVOT (
                    SUM(FUND_WGT_SUM)
                    FOR SUB_FUND_TYP IN ('액티브주식형' AS 액티브주식형, '인덱스주식형' AS 인덱스주식형)
                    )
            ORDER BY PTF_CD
    '''.format(enddate=enddate)


    df = db_connect(df_sql)
    df = df[df['일자']!=df['일자'].min()]

    df['펀드명'] = convert_name(df['펀드명']) # 펀드명 간소화
    # BM이 코스닥인 경우 BM비중 무관하게 투자비중*일수익률
    df["일기여도"] = np.where(df["BM명"]=="코스피", (df["보유비중"]-df["BM비중"])*df['일수익률']*100, df["보유비중"]*df['일수익률']*100) 

    column_order = ['ESG1호', '4-3호','4-5호','4-6호','4-7호','4-8호','4-9호','4-10호','4-11호', 'ESG6호', 'ESG7호']
    corp_list = ['마이다스','브이아이','BNK','안다','베어링','이스트','한투밸류','우리','한투신탁', '하나', '키움']

    # 펀드명 & 운용사명 mapping
    dict_corp_name = {}
    for x, y in zip(column_order, corp_list):
        case = {x:y}
        dict_corp_name.update(case)
  

    # 최근일자 보유비중 
    temp = df[df['일자']==df['일자'].max()]
    temp = temp.pivot_table(index='종목명', columns='펀드명', values='보유비중',fill_value=0).round(2)
    temp = temp[column_order]

    def group_by_sector(bm, data, clas, order=column_order):
        pfo_status = bm.join(data).dropna(subset=order, how='all', axis=0)
        pfo_status_sector = pfo_status.groupby(by=clas)[order].sum()
        kospi_sector = bm[bm['BM명']=='코스피'].groupby(by=clas)[['BM비중']].sum()
        pfo_status_sector = kospi_sector.join(pfo_status_sector)
        pfo_status_sector = pfo_status_sector.sort_values(by='BM비중', ascending=False).fillna(0)
        return pfo_status, pfo_status_sector
    
    pfo_status, pfo_status_sector = group_by_sector(bm_df_lastday, temp, clas='대분류', order=column_order)
    _, pfo_status_sector_lvl2 = group_by_sector(bm_df_lastday, temp, clas='중분류', order=column_order)
    _, pfo_status_sector_lvl3 = group_by_sector(bm_df_lastday, temp, clas='소분류', order=column_order)

    market_df = db_connect(mkt_index_sql).set_index('WKDATE')
    len_idx = len(market_df['KFNAME'].unique())

    if len_idx == 4:
        kosdaq_ret, kospi_ret, k200_ret, k200esg_ret = market_df.sort_values(by="WKDATE").groupby("KFNAME").apply(calculate_period_return)
        (kosdaq_st_index, kosdaq_index), (kospi_st_index, kospi_index),\
            (k200_st_index, k200_index), (k200esg_st_index, k200esg_index) = market_df.sort_values(by="WKDATE").groupby("KFNAME").apply(get_period_value)
    else:
        vu_ret, kosdaq_ret, kospi_ret, k200_ret, k200esg_ret = market_df.sort_values(by="WKDATE").groupby("KFNAME").apply(calculate_period_return)
        (vu_st_index, vu_index), (kosdaq_st_index, kosdaq_index), (kospi_st_index, kospi_index),\
            (k200_st_index, k200_index), (k200esg_st_index, k200esg_index) = market_df.sort_values(by="WKDATE").groupby("KFNAME").apply(get_period_value)
        
    


    # 펀드별 보유현금 및 현금 기여도
    # cash_status = pd.DataFrame(100-temp.sum(axis=0), columns=['현금']).T
    cash_status = (100 - df.groupby(['일자','펀드명']).agg({'보유비중':'sum'}).unstack(level=0)).T
    cash_status = cash_status[column_order]

    ctb_cash_temp = (cash_status * (0 - kospi_ret))/100
    ctb_cash = pd.DataFrame(ctb_cash_temp.sum(axis=0), columns=['현금']).loc[column_order].T
    ctb_cash = ctb_cash.rename(columns=dict_corp_name)

    cash_status = pd.DataFrame(cash_status.iloc[-1, :]).round(2).droplevel(1, axis=1).T
    cash_status.index = ['현금']

    pfo_status_sector = pd.concat([pfo_status_sector, cash_status], axis=0).fillna(0)
    pfo_status_sector_lvl2 = pd.concat([pfo_status_sector_lvl2, cash_status], axis=0).fillna(0)
    pfo_status_sector_lvl3 = pd.concat([pfo_status_sector_lvl3, cash_status], axis=0).fillna(0)


    bm_df_temp = bm_df[bm_df['일자']!=bm_df['일자'].min()].reset_index(drop=True)
    not_in_pfo = [x for x in bm_df_lastday.index if x not in pfo_status.index] # 어떤 펀드도 가지고 있지 않은 종목들
    temp = bm_df_temp[bm_df_temp['종목명'].isin(not_in_pfo)].reset_index(drop=True) # 그런 종목들의 BM 비중과 일 수익률 추출
    temp = temp[temp['BM명']=='코스피'] # 그 중 코스피 속한 종목만
    temp['일기여도'] = (0 - temp['BM비중']) * temp['일수익률'] * 100    # 펀드에서 갖고있진 않지만 코스피 상장 종목 > 미보유한 만큼 수익률 기여
                                                                    # 코스닥 종목 중 미보유는 BM 대비 수익률에 기여 0 이니까

    ctb_others = temp.pivot_table(index='종목명', columns='일자', values='일기여도').sum(axis=1)
    ctb_others = pd.DataFrame(np.array([ctb_others]*len(column_order)).T, columns=column_order, index= ctb_others.index)

    a = pfo_status[pfo_status['BM명']=='코스피'][column_order]
    in_pfo_zero = list(a[a.min(axis=1)==0].index)

    temp = bm_df_temp[bm_df_temp['종목명'].isin(in_pfo_zero)].reset_index(drop=True)
    temp = temp[temp['BM명']=='코스피']
    temp['일기여도'] = (0 - temp['BM비중']) * temp['일수익률'] * 100
    ctb_others2 = temp.pivot_table(index='종목명', columns='일자', values='일기여도').sum(axis=1)
    ctb_others2 = pd.DataFrame(ctb_others2, columns=['일기여도'])


    ctb_by_stock = df.pivot_table(index='종목명', columns='펀드명', values='일기여도', aggfunc='sum', fill_value=0) 
    ctb_by_stock = pd.concat([ctb_by_stock, ctb_others], axis=0)
    ctb_by_stock = ctb_by_stock[column_order]
    sorted_index = [idx for idx in bm_df_lastday.index if idx in ctb_by_stock.index]
    ctb_by_stock = ctb_by_stock.join(bm_df_lastday, how='left')
    ctb_by_stock = ctb_by_stock[list(bm_df_lastday.columns)+column_order]
    ctb_by_stock = ctb_by_stock.loc[sorted_index].round(2)

    b = ctb_by_stock[ctb_by_stock['BM명']=='코스피']
    c = b.merge(ctb_others2, left_on=b.index, right_on=ctb_others2.index, suffixes=("","_B"))

    for col in column_order:
        c[col] = c.apply(
                                lambda row: row["일기여도"] if row[col] == 0 else row[col],
                                axis=1
                                )
    c = c.set_index('key_0').drop(columns=['일기여도']).round(2)
    c.index.name = '종목명'
    ctb_by_stock.update(c)

    ctb_by_stock.rename(columns=dict_corp_name, inplace=True)

    ctb_by_sector = ctb_by_stock.groupby(by='대분류')[corp_list].sum().round(2)
    ctb_by_sector = pd.concat([ctb_by_sector, ctb_cash], axis=0)

    top_funds = pd.DataFrame(ctb_by_sector.sum(axis=0), columns=['기여도합(bp)']).sort_values(by='기여도합(bp)', ascending=False).round(2)
    top_funds.index.name = '운용사'



    # df_funds = db_connect(fund_sql)
    # t_fund_dict = dict()
    # case = {
    #         '308604':'투자풀통합38호',
    #         '308609':'투자풀통합60호',
    #         '308611':'투자풀통합69호',
    #         '308614':'투자풀통합ESG81호',
    #         '308615':'투자풀통합86호',
    #         '308620':'투자풀통합99호',
    #         '308626':'투자풀통합116호',
    #         '308627':'투자풀통합117호',
    #         '308645':'투자풀통합ESG2호',
    #         '308650':'투자풀통합138호',
    #         '308671':'투자풀통합157호',
    #         '308703':'투자풀통합187호',
    #         '308644':'투자풀통합135호',
    #         '308648':'투자풀통합ESG4호',
    #         '308673':'투자풀통합159호',
    #         '308683':'투자풀통합168호',
    #         '308691':'투자풀통합174호'
    #         }
    # t_fund_dict.update(case)
    # df_funds['PTF_CD'] = df_funds['PTF_CD'].map(t_fund_dict)
    # df_funds = df_funds.fillna(0)
    # df_funds['액티브주식형'] = df_funds['액티브주식형']/(df_funds['액티브주식형']+df_funds['인덱스주식형'])
    # df_funds['인덱스주식형'] = 1 - df_funds['액티브주식형']
    # df_funds = df_funds[df_funds['액티브주식형']!=0]
    # df_funds.set_index('PTF_CD', inplace=True)

    
    tab1, tab2 = st.tabs(['현황', '기여도'])
    
    with tab1:

        st.subheader('Market Index')
        cols = st.columns(len_idx)

        cols[0].metric('KOSPI', f'{kospi_index}', delta=f'{kospi_index-kospi_st_index:.2f}, {kospi_ret:.2f}%')
        cols[1].metric('KOSPI 200', f'{k200_index}', delta=f'{k200_index-k200_st_index:.2f}, {k200_ret:.2f}%')
        cols[2].metric('KOSDAQ', f'{kosdaq_index}', delta=f'{kosdaq_index-kosdaq_st_index:.2f}, {kosdaq_ret:.2f}%')
        cols[3].metric('KOSPI 200 ESG', f'{k200esg_index}', delta=f'{k200esg_index-k200esg_st_index:.2f}, {k200esg_ret:.2f}%')
        if len_idx == 5:
            cols[4].metric('KOREA VALUE UP', f'{vu_index}', delta=f'{vu_index-vu_st_index:.2f}, {vu_ret:.2f}%')

        st.divider()

        st.subheader(f'개별펀드 포트폴리오 현황')
        st.markdown(
                        """
                        <p style="font-size:14px; color:#FFFFFF;">
                            <em>* {enddate} 기준</em>
                        </p>
                        """.format(enddate=enddate),
                        unsafe_allow_html=True
                    )
        _, bc = st.columns([9.2, 0.8])
        bc.download_button("Get Data", data=pfo_status.to_csv().encode('cp949'), file_name="Fund_Portfolio.csv", use_container_width=True)
        st.dataframe(pfo_status, height=400, use_container_width=True)
        st.markdown(
                        """
                        <p style="font-size:15px; color:#C2AC97;">
                            <em>* 6호,7호는 '24.11.25.부로 BM 변경되어 이전 기간 기여도 산출 어려움</em>
                        </p>
                        """,
                        unsafe_allow_html=True
                    )
        
        st.divider()

        col1, col2 = st.columns([6, 4])
        col1.subheader('섹터별 비중')
        col1.dataframe(pfo_status_sector, height=425, use_container_width=True)

        col3, col4 = st.columns([5, 5])    
        with col3:
            with st.expander("GICS_LEVEL 2"):    
                st.dataframe(pfo_status_sector_lvl2, height=425, use_container_width=True)
        with col4:
            with st.expander("GICS_LEVEL 3"):
                st.dataframe(pfo_status_sector_lvl3, height=425, use_container_width=True)


        col2.subheader('BM 대비 비중')
        option = col2.selectbox('Fund', column_order)           

        temp = pfo_status_sector[option] - pfo_status_sector['BM비중']
        temp.index.name = '대분류'
        temp = pd.DataFrame(temp, columns=['차이']).reset_index().iloc[:-1,:]
        
        
        chart_temp = alt.Chart(temp).mark_bar().encode(
            x = alt.X('차이:Q', axis=alt.Axis(title='BM 대비(%p)', grid=False)),
            y = alt.Y('대분류:N', axis=alt.Axis(title='섹터', grid=False), sort='-x'),
            color = alt.Color(
                        '차이', 
                        scale=alt.Scale(scheme='blues'),
                        legend=None
                    )
        ).properties(height=330)
        col2.altair_chart(chart_temp, use_container_width=True)

        # st.divider()

        # cols = st.columns(4)
        # for idx, row in enumerate(df_funds.iterrows()):
        #     # 데이터 준비
        #     pie_data = pd.DataFrame({
        #         "Category": ["액티브주식형", "인덱스주식형"],
        #         "Value": [row[1]["액티브주식형"], row[1]["인덱스주식형"]]
        #     })
            
        #     # Altair 파이 차트 생성
        #     chart = alt.Chart(pie_data).mark_arc(outerRadius=100).encode(
        #         theta=alt.Theta(field="Value", type="quantitative"),
        #         color=alt.Color(field="Category", type="nominal", 
        #                         scale=alt.Scale(scheme="tableau10"), legend=None),
        #         tooltip=[alt.Tooltip(field="Category", type="nominal"),
        #                 alt.Tooltip(field="Value", type="quantitative", format=".2%")]
        #     ).properties(
        #                 title=row[0]
        #                 )

        #     # 각 컬럼에 차트 표시
        #     text = alt.Chart(pie_data).mark_text(radius=45, size=14, align="left").encode(
        #             theta=alt.Theta(field="Value", type="quantitative"),
        #             text=alt.Text(field="Value", type="quantitative", format=".2%"),
        #             color=alt.value("black")
        #         )

        #     # 적절한 컬럼에 차트 표시
        #     cols[idx % 4].altair_chart(chart + text, use_container_width=True)
        #     if (idx + 1) % 4 == 0:
        #         cols = st.columns(4)




    with tab2:
        st.subheader('종목별 수익률 기여도')
        _, bc = st.columns([9.2, 0.8])
        bc.download_button("Get Data", data=ctb_by_stock.to_csv().encode('cp949'), file_name="Fund_Contribution.csv", use_container_width=True)
        st.dataframe(ctb_by_stock.style.format(precision=2), height=300, use_container_width=True)
        

        col1, col2 = st.columns([8, 2])

        with col1:
            st.subheader('업종별 수익률 기여도')
            # st.dataframe(ctb_by_sector.style.highlight_max(axis=0, color='#C9E6F0').highlight_min(axis=0, color='#FFE3E3').format(precision=2), height=425, use_container_width=True)
            st.dataframe(ctb_by_sector.style.highlight_max(axis=0, color=color_map[0]).highlight_min(axis=0, color=color_map[1]).format(precision=2), height=425, use_container_width=True)
            
        with col2:
            st.subheader('Top Funds')
            st.dataframe(top_funds, height=425, use_container_width=True)

        # st.divider()


        # temp = ctb_by_sector[['마이다스']]
        # temp.index.name = '섹터'
        # temp = temp.reset_index()


        # chart_temp = alt.Chart(temp).mark_bar().encode(
        #     x = alt.X('섹터:N', axis=alt.Axis(title='섹터', grid=False), sort='-y'),
        #     y = alt.Y('마이다스:Q', axis=alt.Axis(title='기여도', grid=False))
        # ).properties(width=1000, height=330)
        # st.altair_chart(chart_temp)
        

if __name__ == '__main__' :
    main()

