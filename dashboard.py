# 스트림릿 라이브러리를 사용하기 위한 임포트
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import altair as alt
import oracledb as od

color_map = ['#F58220', '#043B72','#00A9CE', '#F0B26B', '#8DC8E8','#CB6015','#AE634E', '#84888B','#7EA0C3', '#C2AC97', '#0086B8']

st.set_page_config(page_title='Invest Pool Dashboard', layout='wide')
today = datetime.today().strftime('%Y-%m-%d')

db_secrets = st.secrets["m_db"]
username = db_secrets["username"]
password = db_secrets["password"]
host = db_secrets["host"]
port = db_secrets["port"]
service_name = db_secrets["service_name"]

@st.cache_data
def db_connect(sql, username=username, password=password, host=host, port=port, service_name=service_name):

    dsn = od.makedsn(host, port, service_name=service_name)
    with od.connect(user=username, password=password, dsn=dsn) as connection:
        # 커서 생성 및 쿼리 실행
        with connection.cursor() as cursor:
            cursor.execute(sql)
            
            # 결과 가져오기
            columns = [col[0] for col in cursor.description]  # 컬럼 이름 가져오기
            rows = cursor.fetchall()  # 모든 데이터 가져오기

            df = pd.DataFrame(rows, columns=columns)
        
    return df


def calculate_period_return(group):
    start_value = group.iloc[0]["CLOSE_INDEX"]
    end_value = group.iloc[-1]["CLOSE_INDEX"]
    return np.round((end_value - start_value)/ start_value * 100,6)  # 수익률 (%)

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
            WHERE BASE_DT = TO_DATE('{targetdate}', 'YYYY-MM-DD')
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
                            <em>Prototype v0.2.1</em>
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
                    WHERE WKDATE BETWEEN TO_DATE('{startdate}', 'YYYY-MM-DD') AND TO_DATE('{enddate}', 'YYYY-MM-DD')
    '''.format(startdate=startdate, enddate=enddate)
    
    bm_sql = '''
            SELECT 일자, 종목명, 대분류, 중분류, 소분류, BM명
                , ROUND(INDEX_MARKET_CAP*100/SUM(INDEX_MARKET_CAP) OVER (PARTITION BY 일자, BM명), 6) AS BM비중
                , 일수익률
            FROM (
                    SELECT B.WKDATE AS 일자
                        , B.JNAME AS 종목명
                        , B.INDUSTRY_LEV1_NM AS 대분류
                        , B.INDUSTRY_LEV2_NM AS 중분류
                        , B.INDUSTRY_LEV3_NM AS 소분류
                        , A.INDEX_NAME_KR AS BM명 
                        , A.INDEX_WEIGHT AS BM비중
                        , A.INDEX_MARKET_CAP
                        , ROUND(B.RATE/100, 6) AS 일수익률

                    FROM AMAKT.E_MA_KRX_PKG_CONST A

                    LEFT JOIN AMAKT.E_MA_FN_JONGMOK_INFO B
                        ON A.FILE_DATE = B.WKDATE
                        AND A.CONSTITUENT_ISIN = B.STDJONG
                        AND B.INDEX_ID IN ('I.001', 'I.201')
                    WHERE A.FILE_DATE BETWEEN TO_DATE('{startdate}', 'YYYY-MM-DD') AND TO_DATE('{enddate}', 'YYYY-MM-DD')
                    AND A.INDEX_CODE1 IN ('1', '2')  -- 1:코스피, 2:코스닥
                    AND A.INDEX_CODE2 IN ('001')  -- 001:코스피, 029:코스피200
            )
            ORDER BY BM명 DESC, BM비중 DESC
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
						WHERE A.FILE_DATE BETWEEN TO_DATE('{startdate}', 'YYYY-MM-DD') AND TO_DATE('{enddate}', 'YYYY-MM-DD')
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
			                                                                        WHERE        BASE_DT BETWEEN TO_DATE('{startdate}', 'YYYY-MM-DD') AND TO_DATE('{enddate}', 'YYYY-MM-DD')
			                                                                            AND      DECMP_TCD = 'U'
			                                                                            AND      AST_CD IN ('SFD', 'FND')
			                                                                            AND      PTF_CD = '308611'
			                                                                            AND		 ISIN_CD IN ('KRZ502465730','KRZ502211090','KRZ502465070'
																												,'KRZ502503770','KRZ502511340','KRZ502514660'
																												,'KRZ502515020','KRZ502589850','KRZ502593290')
																					UNION ALL
																					
																					SELECT   BASE_DT, PTF_CD, ISIN_CD AS SUB_PTF_CD , EVL_AMT , WGT 
			                                                                        FROM     AIVSTP.FSCD_PTF_ENTY_EVL_COMP e1
			                                                                        WHERE        BASE_DT BETWEEN TO_DATE('{startdate}', 'YYYY-MM-DD') AND TO_DATE('{enddate}', 'YYYY-MM-DD')
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
    temp = temp.pivot_table(index='종목명', columns='펀드명', values='보유비중',fill_value=0).round(4)
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
        
    
    sectors = pfo_status[['대분류','중분류','소분류']]
    


    # 펀드별 보유현금 및 현금 기여도
    # cash_status = pd.DataFrame(100-temp.sum(axis=0), columns=['현금']).T
    cash_status = (100 - df.groupby(['일자','펀드명']).agg({'보유비중':'sum'}).unstack(level=0)).T
    cash_status = cash_status[column_order]

    ctb_cash_temp = (cash_status * (0 - kospi_ret))/100
    ctb_cash = pd.DataFrame(ctb_cash_temp.sum(axis=0), columns=['현금']).loc[column_order].T
    ctb_cash = ctb_cash.rename(columns=dict_corp_name)

    cash_status = pd.DataFrame(cash_status.iloc[-1, :]).round(4).droplevel(1, axis=1).T
    cash_status.index = ['현금']

    pfo_status_sector = pd.concat([pfo_status_sector, cash_status], axis=0).fillna(0)
    pfo_status_sector_lvl2 = pd.concat([pfo_status_sector_lvl2, cash_status], axis=0).fillna(0)
    pfo_status_sector_lvl3 = pd.concat([pfo_status_sector_lvl3, cash_status], axis=0).fillna(0)


    ########################################################################################################################################
    
    bm_temp = bm_df.copy()
    # 코스피 종목 중 미보유 종목에 대해 일자별 기여도 산출 
    bm_temp['BM기여도'] = np.where(bm_temp["BM명"]=="코스피", (0-bm_temp["BM비중"])*bm_temp['일수익률']*100, 0) 
    bm_temp = bm_temp.pivot_table(index='종목명', columns='일자', values='BM기여도')

    # 코스피에 있는 종목 중 포트폴리오에 없는 경우 bm기여도 채우기
    result = {}
    for fund in column_order:
        temp = df[df['펀드명']==fund].pivot_table(index='종목명', columns='일자', values='일기여도')
        result[fund] = temp.where(temp.notna(), bm_temp)  
        
    # 기여도 합으로 Dataframe 생성
    ctb_by_stock = bm_df_lastday.copy()
    for fund in column_order:
        a = pd.DataFrame(result[fund].sum(axis=1), columns=[f"{fund}"])
        ctb_by_stock = ctb_by_stock.join(a)

    # 코스닥 종목의 NaN 값은 0으로 채우기 (보유하지 않은 종목이므로)
    ctb_by_stock.loc[ctb_by_stock['BM명']=='코스닥', column_order] = ctb_by_stock.loc[ctb_by_stock['BM명']=='코스닥', column_order].fillna(0)

    # 어디에도 속하지 않은 종목들은 코스피의 경우 BM기여도 채우기
    bm_df_temp = bm_df[bm_df['일자']!=bm_df['일자'].min()].reset_index(drop=True)
    not_in_pfo = ctb_by_stock[ctb_by_stock[column_order].sum(axis=1)==0].index
    ret_others = bm_df_temp[bm_df_temp['종목명'].isin(not_in_pfo)].reset_index(drop=True)
    ret_others['일기여도'] = np.where(ret_others['BM명']=='코스피', (0 - ret_others['BM비중']) * ret_others['일수익률'] * 100, 0)
    ctb_others = ret_others.pivot_table(index='종목명', columns='일자', values='일기여도').sum(axis=1)
    ctb_others = pd.DataFrame(np.array([ctb_others]*len(column_order)).T, columns=column_order, index= ctb_others.index)

    ctb_by_stock.update(ctb_others)

    # 다른 펀드에는 있지만 일부 펀드는 한번도 보유하지 않은 종목
    in_pfo_zero = ctb_by_stock[ctb_by_stock.isna().any(axis=1)].index
    ret_others = bm_df_temp[bm_df_temp['종목명'].isin(in_pfo_zero)].reset_index(drop=True)
    ret_others['일기여도'] = np.where(ret_others['BM명']=='코스피', (0 - ret_others['BM비중']) * ret_others['일수익률'] * 100, 0)

    ctb_others2 = ret_others.pivot_table(index='종목명', columns='일자', values='일기여도').sum(axis=1)
    ctb_others2 = pd.DataFrame(ctb_others2, columns=['일기여도'])

    b = ctb_by_stock[ctb_by_stock['BM명']=='코스피']
    c = b.merge(ctb_others2, left_on=b.index, right_on=ctb_others2.index, suffixes=("","_B"))
    for col in column_order:
        c[col] = c.apply(
                        lambda row: row["일기여도"] if pd.isna(row[col]) else row[col],
                        axis=1
                        )

    c = c.set_index('key_0').drop(columns=['일기여도']).round(4)
    c.index.name = '종목명'
    ctb_by_stock.update(c)
    ctb_by_stock.rename(columns=dict_corp_name, inplace=True)




    ctb_by_sector = ctb_by_stock.groupby(by='대분류')[corp_list].sum().round(4)
    ctb_by_sector = pd.concat([ctb_by_sector, ctb_cash], axis=0)

    ctb_by_sector_lvl2 = ctb_by_stock.groupby(by='중분류')[corp_list].sum().round(4)
    ctb_by_sector_lvl2 = pd.concat([ctb_by_sector_lvl2, ctb_cash], axis=0)

    ctb_by_sector_lvl3 = ctb_by_stock.groupby(by='소분류')[corp_list].sum().round(4)
    ctb_by_sector_lvl3 = pd.concat([ctb_by_sector_lvl3, ctb_cash], axis=0)


    top_funds = pd.DataFrame(ctb_by_sector.sum(axis=0), columns=['기여도합(bp)']).sort_values(by='기여도합(bp)', ascending=False).round(4)
    top_funds.index.name = '운용사'

    
    tab1, tab2 , tab3= st.tabs(['현황', '기여도', '통합펀드'])
    
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
        col1, _, col3 = st.columns([0.8, 8.4, 0.8])
        col3.download_button("Get Data", data=pfo_status.to_csv().encode('cp949'), file_name="Fund_Portfolio.csv", use_container_width=True)
        
        pfo_status_vs_bm = pfo_status[column_order] - pfo_status[['BM비중']].values    
        pfo_status_vs_bm = pd.concat([pfo_status.iloc[:,:6], pfo_status_vs_bm], axis=1)

        comparison = col1.checkbox("BM 대비")

        if comparison:
            pfo_data = pfo_status_vs_bm
        else:    
            pfo_data = pfo_status

        st.dataframe(pfo_data.style.format(precision=2),\
                      height=400, use_container_width=True)
        

        st.markdown(
                        """
                        <p style="font-size:15px; color:#C2AC97;">
                            <em>* 6호,7호는 '24.11.25.부로 BM 변경되어 이전 기간 기여도 산출 어려움</em>
                        </p>
                        """,
                        unsafe_allow_html=True
                    )
        
        st.divider()

        # tt = pfo_status.copy()
        # tt['BM수익률'] = tt['BM비중'] * tt['일수익률']
        # tt2 = tt.groupby('대분류').agg({'BM수익률':'sum'})
        # st.dataframe(tt2)

        col1, col2 = st.columns([6, 4])
        col1.subheader('섹터별 비중')
        col1.dataframe(pfo_status_sector.style.format(precision=2), height=425, use_container_width=True)

        col3, col4 = st.columns([5, 5])    
        with col3:
            with st.expander("GICS_LEVEL 2"):    
                st.dataframe(pfo_status_sector_lvl2.style.format(precision=2), height=425, use_container_width=True)
        with col4:
            with st.expander("GICS_LEVEL 3"):
                st.dataframe(pfo_status_sector_lvl3.style.format(precision=2), height=425, use_container_width=True)


        col2.subheader('BM 대비 비중(펀드별)')
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

        st.divider()

        col1, col2 = st.columns([7, 3])
        with col1:
            st.subheader('섹터별 비중(Chart)')
            options = list(pfo_status_sector.index)
            selection = st.segmented_control(
                '대분류', options, selection_mode='single'
            )

            if selection:
                options_lvl2 = list(sectors[sectors['대분류']==selection]['중분류'].unique())
                selection_lvl2 = st.segmented_control(
                    '중분류', options_lvl2, selection_mode='single'
                )

                if selection_lvl2:
                    selected = selection_lvl2
                    df_sector = pfo_status_sector_lvl2

                    options_lvl3 = list(sectors[sectors['중분류']==selection_lvl2]['소분류'].unique())
                    selection_lvl3 = st.segmented_control(
                        '소분류', options_lvl3, selection_mode='single'
                    )

                    if selection_lvl3:
                        selected = selection_lvl3
                        df_sector = pfo_status_sector_lvl3

                else:
                    selected = selection
                    df_sector = pfo_status_sector
                
                result = pd.DataFrame(df_sector.loc[selected]).sort_values(by=selected, ascending=False)
                
                result = result.rename(index={'BM비중':'KOSPI'}, columns={selected:'비중'})
                result.index.name = '펀드'
                result = result.reset_index()

                chart_temp = alt.Chart(result).mark_bar().encode(
                y = alt.Y('비중:Q', axis=alt.Axis(title='비중', grid=False)),
                x = alt.X('펀드:N', axis=alt.Axis(title='펀드', grid=False, labelAngle=-45), sort='-y'),
                color = alt.condition(
                            "datum.펀드 == 'KOSPI'",
                            alt.value(color_map[0]),
                            alt.value(color_map[1])
                        )
                ).properties(height=330, width=800).configure_axis(labelFontSize=14, titleFontSize=14)
                st.altair_chart(chart_temp)

        with col2:
            st.subheader('개별펀드 코스닥 비중')
            market_wgt = pfo_status.groupby('BM명')[column_order].sum().T
            market_wgt.index.name = '개별펀드'
            market_wgt.reset_index(inplace=True)
            market_wgt = market_wgt[['개별펀드','코스닥']]
            market_wgt = market_wgt.sort_values('코스닥', ascending=False)
            # st.dataframe(market_wgt)
            melted_df = market_wgt.melt(id_vars='개별펀드', var_name='코스닥', value_name='비율')

            
            pie_chart = alt.Chart(melted_df).mark_arc(innerRadius=50).encode(
                                theta=alt.Theta(field='비율', type='quantitative'),  # 비율 데이터
                                color = alt.Color(
                                            field='개별펀드',
                                            type='nominal',
                                            scale=alt.Scale(range=color_map),
                                            sort=alt.EncodingSortField(
                                                    field='비율',  # 비율 기준으로 정렬
                                                    order='descending'  # 내림차순
                                                ),
                                        ),
                                order=alt.Order(field='비율', type='quantitative', sort='descending')
                            ).properties(
                                width=330,
                                height=330
                            )

            st.altair_chart(pie_chart)
        

    with tab2:
        st.subheader('종목별 수익률 기여도')
        _, bc = st.columns([9.2, 0.8])
        bc.download_button("Get Data", data=ctb_by_stock.to_csv().encode('cp949'), file_name="Fund_Contribution.csv", use_container_width=True)
        st.dataframe(ctb_by_stock.style.format(precision=2), height=300, use_container_width=True)
        

        col1, col2 = st.columns([8, 2])

        with col1:
            st.subheader('업종별 수익률 기여도')
            st.dataframe(ctb_by_sector.style.highlight_max(axis=0, color='#C9E6F0').highlight_min(axis=0, color='#FFE3E3').format(precision=2),\
                             height=425, use_container_width=True)
            # st.dataframe(ctb_by_sector.style.highlight_max(axis=0, color=color_map[0]).highlight_min(axis=0, color=color_map[1]).format(precision=2),\
            #               height=425, use_container_width=True)

        with col2:
            st.subheader('Top Funds')
            st.dataframe(top_funds.style.format(precision=2), height=425, use_container_width=True)

        st.divider()

        # col1, _ = st.columns([1, 9])
        # option = col1.selectbox('Sector Level', ['중분류', '소분류'])

        st.subheader('See Details')
        options = ['중분류','소분류']
        selection = st.segmented_control(
            'Sector Level', options, selection_mode='single'
        )

        if selection == '중분류':
            st.dataframe(ctb_by_sector_lvl2.style.format(precision=2).background_gradient(cmap='YlGnBu'),height=600,use_container_width=True)
        elif selection == '소분류':
            st.dataframe(ctb_by_sector_lvl3.style.format(precision=2).background_gradient(cmap='YlGnBu'),height=800,use_container_width=True)

    with tab3:
        
        st.subheader('통합펀드 내 개별펀드 비중')

        ks_fund_dict = {'308620':'투자풀통합99호', '308626':'투자풀통합116호', '308703':'투자풀통합187호', '308604':'투자풀통합38호', '308611':'투자풀통합69호',
                        '308614':'투자풀통합ESG81호', '308615':'투자풀통합86호', '308627':'투자풀통합117호', '308645':'투자풀통합ESG2호','308650':'투자풀통합138호',
                        '308671':'투자풀통합157호'}

        col1, col2, _ = st.columns([2, 2, 6])       
        options = col1.selectbox("Selected Fund", list(ks_fund_dict.values()), index=None)
        
        if not options: 
            pass
        else:
            targetfund = [key for key, value in ks_fund_dict.items() if value == options][0]
    
            total_df_sql = '''
                    WITH KOSPI_SECTOR AS (
                        SELECT
                        WKDATE, INDEX_ID, STDJONG, JNAME, INDUSTRY_LEV1_NM, INDUSTRY_LEV2_NM, INDUSTRY_LEV3_NM
                        FROM E_MA_FN_JONGMOK_INFO
                        WHERE INDEX_ID IN ('I.001', 'I.201') --코스피
                        AND WKDATE = TO_DATE('{targetdate}', 'YYYY-MM-DD')
                        )
                    , RAWDATA AS (		
                                    
                            SELECT     A.BASE_DT
                                    , A.PTF_CD
                                    , CASE WHEN INSTR( V.F_NM ,'혼합') > 0   THEN 1    -- 혼합형이면 1
                                        ELSE 0                                      -- 혼합형 아니면 0
                                    END AS FLAG
                                    , A.SUB_FUND_CD
                                    , A.FUND_WGT                    , A.FUND_NM
                                    , A.ISIN_CD
                                    , A.KOR_NM
                                    , KS.INDUSTRY_LEV1_NM
                                    , KS.INDUSTRY_LEV2_NM
                                    , KS.INDUSTRY_LEV3_NM
                                    , A.SEC_WGT
                                    , ROUND(A.FUND_WGT * A.SEC_WGT / 100, 4) AS RESULT
                                                
                                    FROM       
                                                (  
                                                    SELECT   a.BASE_DT
                                                            , a.PTF_CD
                                                            , a.SUB_FUND_CD
                                                            , a.FUND_NM
                                                            , a.ISIN_CD
                                                            , a.KOR_NM
                                                            , a.FUND_NAV
                                                            , a.FUND_WGT*100 AS FUND_WGT
                                                            , a.SEC_WGT*100 AS SEC_WGT
                                                            , NVL(SUBSTR(io.CLAS_CD, 1, 3), 'W00') AS GICS_LVL1
                                                            , NVL(SUBSTR(io.CLAS_CD, 1, 5), 'W0000') AS GICS_LVL2
                                                            , NVL(io.CLAS_CD, 'W000000') AS GICS_LVL3
                                                    FROM    (       
                                                            SELECT  BASE_DT, PTF_CD, SUB_FUND_CD, FUND_NM, ISIN_CD , KOR_NM, FUND_NAV, ROUND(FUND_WGT, 8) AS FUND_WGT
                                                                    , ROUND(EVL_AMT/FUND_NAV,8) AS SEC_WGT
                                                            FROM    (
                                                                        SELECT  e.BASE_DT,e.PTF_CD, NULL AS SUB_FUND_CD, m.KOR_NM AS FUND_NM
                                                                                , NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) AS ISIN_CD
                                                                                , NVL(sm.KOR_NM, em.KOR_NM) AS KOR_NM                              -- 선물 내 이름 사용 후 없으면 주식 이름 사용
                                                                                , n.NET_AST_TOT_AMT AS FUND_NAV, n.NET_AST_TOT_AMT/n.NET_AST_TOT_AMT AS FUND_WGT 
                                                                                , SUM(e.EVL_AMT * NVL(w.WGT, 1)) AS EVL_AMT                        -- 선물 내 비중이 있으면 곱해주고 그게 아니라면 주식이니 그냥 1 곱합
                                                                        FROM    AIVSTP.FSBD_PTF_MSTR m
                                                                                INNER JOIN          AIVSTP.FSCD_PTF_ENTY_EVL_COMP e
                                                                                            ON      m.PTF_CD = e.PTF_CD 
                                                                                            AND     e.BASE_DT = TO_DATE('{targetdate}', 'YYYY-MM-DD')
                                                                                            AND     e.DECMP_TCD =  'A'                               -- 통합펀드(PAR)의 경우에는 'A'로 분해, 개별펀드는 'E'로 분해
                                                                                            AND     m.EX_TCD = e.EX_TCD                              -- KRW 
                                                                                            AND     e.AST_CD  IN  ('STK', 'SFT')                     -- 주식과 선물 포함, 참고로 'A'로 분해시 ETF까지는 분해되어 있음
                                                                            LEFT OUTER JOIN      AMAKT.FSBD_ERM_STK_MAP_MT em                 -- 삼성전자우 와 같은 종목을 삼성전자로 인식하기 위해 필요
                                                                                                ON  e.ISIN_CD = em.ISIN_CD
                                                                            LEFT OUTER JOIN      AMAKT.FSBD_ERM_IDX_ENTY_WGT w                    -- 선물 분해 로직
                                                                                                ON  NVL(em.RM_ISIN_CD, e.ISIN_CD) = w.IDX_CD      -- 선물의 RM_ISIN_CD 를 활용해서 연결
                                                                                                AND e.BASE_DT = w.BASE_DT 
                                                                            LEFT OUTER JOIN      AMAKT.FSBD_ERM_STK_MAP_MT sm                  -- em 테이블에서 했던 것 동일 반복
                                                                                                ON  w.ISIN_CD = sm.ISIN_CD   
                                                                                LEFT OUTER JOIN     AIVSTP.FSCD_PTF_EVL_COMP n 
                                                                                                ON  e.BASE_DT = n.BASE_DT   
                                                                                                AND m.PTF_CD = n.PTF_CD     
                                                                                                
                                                                        WHERE m.PTF_CD = '{targetfund}'
                                                                        GROUP BY  e.BASE_DT,e.PTF_CD,  m.KOR_NM,  NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) ,  NVL(sm.KOR_NM, em.KOR_NM) , n.NET_AST_TOT_AMT 
                                                                        
                                                                        UNION ALL               -- 개별펀드 포함
                                                                                                                
                                                                        SELECT  e.BASE_DT,k.PTF_CD, k.SUB_PTF_CD , m.KOR_NM AS FUND_NM
                                                                                , NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) AS ISIN_CD
                                                                                , NVL(sm.KOR_NM, em.KOR_NM) AS KOR_NM                             -- 선물 내 이름 사용 후 없으면 주식 이름 사용
                                                                                , n.NET_AST_TOT_AMT AS FUND_NAV , k.WGT AS FUND_WGT
                                                                                , SUM(e.EVL_AMT * NVL(w.WGT, 1)) AS EVL_AMT                        -- 선물 내 비중이 있으면 곱해주고 그게 아니라면 주식이니 그냥 1 곱합
                                                                        FROM    AIVSTP.FSBD_PTF_MSTR m
                                                                                INNER JOIN      (       
                                                                                                    SELECT   BASE_DT, PTF_CD, ISIN_CD AS SUB_PTF_CD , EVL_AMT , WGT 
                                                                                                    FROM     AIVSTP.FSCD_PTF_ENTY_EVL_COMP e1
                                                                                                    WHERE        BASE_DT = TO_DATE('{targetdate}', 'YYYY-MM-DD')
                                                                                                        AND      DECMP_TCD = 'U'
                                                                                                        AND      AST_CD IN ('SFD', 'FND')
                                                                                                        AND      PTF_CD = '{targetfund}'
                                                                                                        
                                                                                                ) k
                                                                                        ON      m.PTF_CD = k.SUB_PTF_CD
                                                                                        AND     m.KOR_NM LIKE '%주식%'
                                                                                LEFT JOIN          AIVSTP.FSCD_PTF_ENTY_EVL_COMP e
                                                                                            ON      k.SUB_PTF_CD = e.PTF_CD 
                                                                                            AND     e.BASE_DT = k.BASE_DT
                                                                                            AND     e.DECMP_TCD =  'E' --  개별펀드는 'E'로 분해
                                                                                            AND     e.AST_CD  IN  ('STK', 'SFT')                     -- 주식과 선물 포함, 참고로 'A'로 분해시 ETF까지는 분해되어 있음         
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
                                                            LEFT OUTER JOIN AMAKT.FSBD_ENTY_CCS_IO_MT io
                                                                    ON  a.BASE_DT BETWEEN io.ST_DT AND io.END_DT 
                                                                    AND NVL(s.RM_ISIN_CD, a.ISIN_CD) = io.ISIN_CD 
                                                                    AND io.CCS_TCD = 'STK' 
                                                                    AND io.CLAS_TYP = 'W3'
                                                            LEFT OUTER JOIN AIVSTP.FSBD_CCS_MSTR c
                                                                    ON io.CLAS_CD = c.CLAS_CD 
                                                                    AND c.CCS_TCD = 'STK' 
                                                                    AND c.CLAS_TYP = 'W'   
                                                            ORDER BY a.PTF_CD, a.FUND_NM , a.ISIN_CD
                                                        ) A
                                    INNER JOIN KOSPI_SECTOR KS
                                        ON A.ISIN_CD = KS.STDJONG                             
                                    LEFT OUTER JOIN      AIVSTP.FSBD_CCS_MSTR B
                                                    ON   B.CCS_TCD = 'STK' 
                                                    AND  B.CLAS_CD = A.GICS_LVL1
                                    LEFT OUTER JOIN      AIVSTP.FSBD_CCS_MSTR C
                                                    ON   C.CCS_TCD = 'STK' 
                                                    AND  C.CLAS_CD = A.GICS_LVL2
                                    LEFT OUTER JOIN      AIVSTP.FSBD_CCS_MSTR D
                                                    ON   D.CCS_TCD = 'STK' 
                                                    AND  D.CLAS_CD = A.GICS_LVL3        
                                    LEFT OUTER JOIN  
                                                    (       
                                                            SELECT  * 
                                                            FROM    AMAKT.FSBD_ENTY_CCS_IO_MT
                                                            WHERE  1=1
                                                            AND   CCS_TCD = 'STK' 
                                                            AND   CLAS_TYP = 'S'
                                                    ) E
                                            ON  A.ISIN_CD = E.ISIN_CD           
                                            AND  A.BASE_DT   BETWEEN E.ST_DT AND E.END_DT 
                                    INNER JOIN           POLSEL.V_FUND_CD  V
                                                    ON   A.PTF_CD = V.AM_FUND_CD    
                                                    WHERE FUND_WGT NOT LIKE '100'
                                )
                                
                SELECT *
                FROM ( SELECT FUND_NM, INDUSTRY_LEV1_NM, ROUND(SEC_WGT/100, 6) AS SEC_WGT
                        FROM RAWDATA
                    )
                PIVOT (
                        SUM(SEC_WGT)
                        FOR INDUSTRY_LEV1_NM IN ('IT' AS IT, '경기소비재' AS 경기소비재, '금융' AS 금융, '산업재' AS 산업재, '소재' AS 소재, 
                                            '에너지' AS 에너지, '유틸리티' AS 유틸리티, '의료' AS 의료, '통신서비스' AS 통신서비스, '필수소비재' AS 필수소비재)
                        )
                ORDER BY FUND_NM ASC

            '''.format(targetdate=enddate, targetfund = targetfund)
            
            total_wgt_df_sql = '''
                WITH KOSPI_SECTOR AS (
                    SELECT
                    WKDATE, INDEX_ID, STDJONG, JNAME, INDUSTRY_LEV1_NM, INDUSTRY_LEV2_NM, INDUSTRY_LEV3_NM
                    FROM E_MA_FN_JONGMOK_INFO
                    WHERE INDEX_ID IN ('I.001', 'I.201') --코스피
                    AND WKDATE = TO_DATE('{targetdate}', 'YYYY-MM-DD')
                    )
                , RAWDATA AS (		
                                
                        SELECT     A.BASE_DT
                                , A.PTF_CD
                                , CASE WHEN INSTR( V.F_NM ,'혼합') > 0   THEN 1    -- 혼합형이면 1
                                    ELSE 0                                      -- 혼합형 아니면 0
                                END AS FLAG
                                , A.SUB_FUND_CD
                                , A.FUND_WGT                    , A.FUND_NM
                                , A.ISIN_CD
                                , A.KOR_NM
                                , KS.INDUSTRY_LEV1_NM
                                , KS.INDUSTRY_LEV2_NM
                                , KS.INDUSTRY_LEV3_NM
                                , A.SEC_WGT
                                , ROUND(A.FUND_WGT * A.SEC_WGT / 100, 4) AS RESULT
                                            
                                FROM       
                                            (  
                                                SELECT   a.BASE_DT
                                                        , a.PTF_CD
                                                        , a.SUB_FUND_CD
                                                        , a.FUND_NM
                                                        , a.ISIN_CD
                                                        , a.KOR_NM
                                                        , a.FUND_NAV
                                                        , a.FUND_WGT*100 AS FUND_WGT
                                                        , a.SEC_WGT*100 AS SEC_WGT
                                                        , NVL(SUBSTR(io.CLAS_CD, 1, 3), 'W00') AS GICS_LVL1
                                                        , NVL(SUBSTR(io.CLAS_CD, 1, 5), 'W0000') AS GICS_LVL2
                                                        , NVL(io.CLAS_CD, 'W000000') AS GICS_LVL3
                                                FROM    (       
                                                        SELECT  BASE_DT, PTF_CD, SUB_FUND_CD, FUND_NM, ISIN_CD , KOR_NM, FUND_NAV, ROUND(FUND_WGT, 8) AS FUND_WGT
                                                                , ROUND(EVL_AMT/FUND_NAV,8) AS SEC_WGT
                                                        FROM    (
                                                                    SELECT  e.BASE_DT,e.PTF_CD, NULL AS SUB_FUND_CD, m.KOR_NM AS FUND_NM
                                                                            , NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) AS ISIN_CD
                                                                            , NVL(sm.KOR_NM, em.KOR_NM) AS KOR_NM                              -- 선물 내 이름 사용 후 없으면 주식 이름 사용
                                                                            , n.NET_AST_TOT_AMT AS FUND_NAV, n.NET_AST_TOT_AMT/n.NET_AST_TOT_AMT AS FUND_WGT 
                                                                            , SUM(e.EVL_AMT * NVL(w.WGT, 1)) AS EVL_AMT                        -- 선물 내 비중이 있으면 곱해주고 그게 아니라면 주식이니 그냥 1 곱합
                                                                    FROM    AIVSTP.FSBD_PTF_MSTR m
                                                                            INNER JOIN          AIVSTP.FSCD_PTF_ENTY_EVL_COMP e
                                                                                        ON      m.PTF_CD = e.PTF_CD 
                                                                                        AND     e.BASE_DT = TO_DATE('{targetdate}', 'YYYY-MM-DD')
                                                                                        AND     e.DECMP_TCD =  'A'                               -- 통합펀드(PAR)의 경우에는 'A'로 분해, 개별펀드는 'E'로 분해
                                                                                        AND     m.EX_TCD = e.EX_TCD                              -- KRW 
                                                                                        AND     e.AST_CD  IN  ('STK', 'SFT')                     -- 주식과 선물 포함, 참고로 'A'로 분해시 ETF까지는 분해되어 있음
                                                                        LEFT OUTER JOIN      AMAKT.FSBD_ERM_STK_MAP_MT em                 -- 삼성전자우 와 같은 종목을 삼성전자로 인식하기 위해 필요
                                                                                            ON  e.ISIN_CD = em.ISIN_CD
                                                                        LEFT OUTER JOIN      AMAKT.FSBD_ERM_IDX_ENTY_WGT w                    -- 선물 분해 로직
                                                                                            ON  NVL(em.RM_ISIN_CD, e.ISIN_CD) = w.IDX_CD      -- 선물의 RM_ISIN_CD 를 활용해서 연결
                                                                                            AND e.BASE_DT = w.BASE_DT 
                                                                        LEFT OUTER JOIN      AMAKT.FSBD_ERM_STK_MAP_MT sm                  -- em 테이블에서 했던 것 동일 반복
                                                                                            ON  w.ISIN_CD = sm.ISIN_CD   
                                                                            LEFT OUTER JOIN     AIVSTP.FSCD_PTF_EVL_COMP n 
                                                                                            ON  e.BASE_DT = n.BASE_DT   
                                                                                            AND m.PTF_CD = n.PTF_CD     
                                                                                            
                                                                    WHERE m.PTF_CD = '{targetfund}'
                                                                    GROUP BY  e.BASE_DT,e.PTF_CD,  m.KOR_NM,  NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) ,  NVL(sm.KOR_NM, em.KOR_NM) , n.NET_AST_TOT_AMT 
                                                                    
                                                                    UNION ALL               -- 개별펀드 포함
                                                                                                            
                                                                    SELECT  e.BASE_DT,k.PTF_CD, k.SUB_PTF_CD , m.KOR_NM AS FUND_NM
                                                                            , NVL(sm.RM_ISIN_CD, NVL(w.ISIN_CD, NVL(em.RM_ISIN_CD, e.ISIN_CD))) AS ISIN_CD
                                                                            , NVL(sm.KOR_NM, em.KOR_NM) AS KOR_NM                             -- 선물 내 이름 사용 후 없으면 주식 이름 사용
                                                                            , n.NET_AST_TOT_AMT AS FUND_NAV , k.WGT AS FUND_WGT
                                                                            , SUM(e.EVL_AMT * NVL(w.WGT, 1)) AS EVL_AMT                        -- 선물 내 비중이 있으면 곱해주고 그게 아니라면 주식이니 그냥 1 곱합
                                                                    FROM    AIVSTP.FSBD_PTF_MSTR m
                                                                            INNER JOIN      (       
                                                                                                SELECT   BASE_DT, PTF_CD, ISIN_CD AS SUB_PTF_CD , EVL_AMT , WGT 
                                                                                                FROM     AIVSTP.FSCD_PTF_ENTY_EVL_COMP e1
                                                                                                WHERE        BASE_DT = TO_DATE('{targetdate}', 'YYYY-MM-DD')
                                                                                                    AND      DECMP_TCD = 'U'
                                                                                                    AND      AST_CD IN ('SFD', 'FND')
                                                                                                    AND      PTF_CD = '{targetfund}'
                                                                                                    
                                                                                            ) k
                                                                                    ON      m.PTF_CD = k.SUB_PTF_CD
                                                                                    AND     m.KOR_NM LIKE '%주식%'
                                                                            LEFT JOIN          AIVSTP.FSCD_PTF_ENTY_EVL_COMP e
                                                                                        ON      k.SUB_PTF_CD = e.PTF_CD 
                                                                                        AND     e.BASE_DT = k.BASE_DT
                                                                                        AND     e.DECMP_TCD =  'E' --  개별펀드는 'E'로 분해
                                                                                        AND     e.AST_CD  IN  ('STK', 'SFT')                     -- 주식과 선물 포함, 참고로 'A'로 분해시 ETF까지는 분해되어 있음         
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
                                                        LEFT OUTER JOIN AMAKT.FSBD_ENTY_CCS_IO_MT io
                                                                ON  a.BASE_DT BETWEEN io.ST_DT AND io.END_DT 
                                                                AND NVL(s.RM_ISIN_CD, a.ISIN_CD) = io.ISIN_CD 
                                                                AND io.CCS_TCD = 'STK' 
                                                                AND io.CLAS_TYP = 'W3'
                                                        LEFT OUTER JOIN AIVSTP.FSBD_CCS_MSTR c
                                                                ON io.CLAS_CD = c.CLAS_CD 
                                                                AND c.CCS_TCD = 'STK' 
                                                                AND c.CLAS_TYP = 'W'   
                                                        ORDER BY a.PTF_CD, a.FUND_NM , a.ISIN_CD
                                                    ) A
                                INNER JOIN KOSPI_SECTOR KS
                                    ON A.ISIN_CD = KS.STDJONG                             
                                LEFT OUTER JOIN      AIVSTP.FSBD_CCS_MSTR B
                                                ON   B.CCS_TCD = 'STK' 
                                                AND  B.CLAS_CD = A.GICS_LVL1
                                LEFT OUTER JOIN      AIVSTP.FSBD_CCS_MSTR C
                                                ON   C.CCS_TCD = 'STK' 
                                                AND  C.CLAS_CD = A.GICS_LVL2
                                LEFT OUTER JOIN      AIVSTP.FSBD_CCS_MSTR D
                                                ON   D.CCS_TCD = 'STK' 
                                                AND  D.CLAS_CD = A.GICS_LVL3        
                                LEFT OUTER JOIN  
                                                (       
                                                        SELECT  * 
                                                        FROM    AMAKT.FSBD_ENTY_CCS_IO_MT
                                                        WHERE  1=1
                                                        AND   CCS_TCD = 'STK' 
                                                        AND   CLAS_TYP = 'S'
                                                ) E
                                        ON  A.ISIN_CD = E.ISIN_CD           
                                        AND  A.BASE_DT   BETWEEN E.ST_DT AND E.END_DT 
                                INNER JOIN           POLSEL.V_FUND_CD  V
                                                ON   A.PTF_CD = V.AM_FUND_CD    
                                                WHERE FUND_WGT NOT LIKE '100'
                            )
                            
            SELECT FUND_NM
                , ROUND(FIRST_VALUE(FUND_WGT) OVER (PARTITION BY FUND_NM)/100, 5) AS FUND_WGT
            FROM RAWDATA
            GROUP BY FUND_NM, FUND_WGT
        '''.format(targetdate=enddate,targetfund = targetfund)


            df_total = db_connect(total_df_sql)
            df_total_wgt = db_connect(total_wgt_df_sql)
            # df_cumret = db_connect(cumret_sql)

            df_total['FUND_NM'] = convert_name(df_total['FUND_NM'])
            df_total_wgt['FUND_NM'] = convert_name(df_total_wgt['FUND_NM'])

            df_total = df_total.set_index('FUND_NM').fillna(0)
            df_total_wgt = df_total_wgt.set_index('FUND_NM').fillna(0)
            
            active_temp = df_total_wgt[df_total_wgt.index.isin(column_order)]
            index_temp = df_total_wgt[~df_total_wgt.index.isin(column_order)]

            active_wgt = np.round(active_temp.to_numpy().sum() / (active_temp.to_numpy().sum()+index_temp.to_numpy().sum()), 6)
            index_wgt = np.round(index_temp.to_numpy().sum() / (active_temp.to_numpy().sum()+index_temp.to_numpy().sum()), 6)
            
            df_total_wgt = df_total_wgt[df_total_wgt.index.isin(column_order)]

            df_total_wgt['펀드내비중(액티브)'] = df_total_wgt['FUND_WGT']/sum(df_total_wgt['FUND_WGT'])
            df_total_wgt = df_total_wgt.drop('FUND_WGT', axis=1).sort_values('펀드내비중(액티브)', ascending=False)

            df_total_wgt.rename(index=dict_corp_name, inplace=True)

            df_total = df_total * 100
            df_total_wgt = df_total_wgt * 100


            col1, col2 = st.columns([8,2])

            col1.dataframe(df_total.style.format(precision=2), use_container_width=True)
            col2.dataframe(df_total_wgt.style.format(precision=2), use_container_width=True)
            
            col2.write(f"Active: {np.round(active_wgt*100,2)}% / Index: {np.round(index_wgt*100,2)}%")

            st.divider()

            col1, col2, col3 = st.columns([7, 1.5, 1.5])
            col1.subheader('통합펀드 내 개별펀드 수익률 기여도')

            df_a = ctb_by_sector.copy().T
            df_b = df_total_wgt.copy()
            
            df_b_aligned = df_a.index.map(df_b['펀드내비중(액티브)'])
            df_result = df_a.mul(df_b_aligned, axis=0).T
            df_result = df_result * active_wgt / 100
            # df_result = df_result.sum(axis=1).sort_values(ascending=False)


            col1.dataframe(df_result.style.highlight_max(axis=0, color='#C9E6F0').highlight_min(axis=0, color='#FFE3E3')\
                           .format(precision=2), use_container_width=True)
            
            col2.subheader('Top Sectors')
            df_result_temp = pd.DataFrame(df_result.sum(axis=1), columns=['기여도합(bp)']).sort_values('기여도합(bp)',ascending=False)
            col2.dataframe(df_result_temp.style.format(precision=2), use_container_width=True)
            
            cont_sum = np.round(np.sum(df_result_temp)[0], 2)
            col2.write(f"총 기여도: {cont_sum}bp")


            col3.subheader('Top Funds')
            df_result_temp = pd.DataFrame(df_result.sum(axis=0), columns=['기여도합(bp)']).sort_values('기여도합(bp)',ascending=False)
            col3.dataframe(df_result_temp.style.format(precision=2), use_container_width=True)
            
            cont_sum = np.round(np.sum(df_result_temp)[0], 2)


            

if __name__ == '__main__' :
    main()

 
