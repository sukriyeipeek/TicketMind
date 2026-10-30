import os
import httpx
import streamlit as st
BASE=os.getenv('TICKETMIND_API_URL','http://127.0.0.1:8000/analyze').removesuffix('/analyze')
def get(path):
    try:
        r=httpx.get(BASE+path,timeout=20,trust_env=False)
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError,ValueError):
        st.info('Start the Banking77 API and train the models using the README commands.')
        return None
st.set_page_config(page_title='TicketMind · Banking77',layout='wide')
st.title('TicketMind · Banking77')
st.caption('English online banking intents · Official PolyAI dataset · 77 categories · CC BY 4.0')
analyze,examples,metrics,provenance=st.tabs(['Analyze','Real training examples','Evaluation','Data provenance'])
with analyze:
    model=st.selectbox('Model',['word','char'])
    with st.form('analyze'):
        text=st.text_area('Banking message',placeholder='Enter an English banking support request.',max_chars=4000)
        submit=st.form_submit_button('Classify',type='primary')
    if submit:
        try:
            r=httpx.post(BASE+'/analyze',json={'text':text,'model':model},timeout=30,trust_env=False)
            if r.is_error: st.error(r.json().get('detail','Request failed'))
            else:
                result=r.json()
                st.subheader((result['category'] or 'Unrecognized vocabulary').replace('_',' '))
                top=sorted(result['category_scores'].items(),key=lambda item:-item[1])[:10]
                st.bar_chart({'Intent':[k.replace('_',' ') for k,v in top],'Model score':[v for k,v in top]},x='Intent',y='Model score')
                st.warning('Human review required. Scores are uncalibrated; no banking transaction is performed.')
        except httpx.HTTPError: st.error('Local API unavailable.')
with examples:
    catalog=get('/labels')
    if catalog:
        label=st.selectbox('Intent',catalog['labels'],format_func=lambda v:v.replace('_',' '))
        records=get('/examples?label='+label)
        if records: st.dataframe(records['examples'],hide_index=True)
    st.caption('These are original training messages, not generated examples. Official test messages are not shown here.')
with metrics:
    report=get('/experiments')
    if report:
        st.dataframe([{'Model':k,'Validation macro F1':v['validation']['macro_f1'],'Validation accuracy':v['validation']['accuracy'],'Fit seconds':v['fit_seconds']} for k,v in report['models'].items()],hide_index=True)
        st.write('Selected using validation only:',report['selected_model'])
        if report['official_test']:
            st.metric('Official test macro F1',f"{report['official_test']['macro_f1']:.3f}")
            st.metric('Official test accuracy',f"{report['official_test']['accuracy']:.3f}")
        st.json(report['audit'])
with provenance:
    manifest=get('/data/manifest')
    if manifest: st.json(manifest)
    st.write('Banking77 contains intent labels, not verified banking policies or named-entity spans. Policy answer generation and the old synthetic NER/RAG evaluations are removed from the active application.')
