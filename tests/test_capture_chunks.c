/* Owning K6 transaction, real bounded Metal capture/archives, synthetic model
 * rows/history/head. Mechanism evidence only, never numerical inference. */
#include "../ds4.h"
static bool chunk_before_rows(void*,uint32_t);
static void chunk_after_rows(void*,const int*,uint32_t);
#define DS41_FIXTURE_BEFORE_ROWS(g,n) chunk_before_rows(g,n)
#define DS41_FIXTURE_AFTER_ROWS(g,v,n) chunk_after_rows(g,v,n)
static bool fixture_body_dispatch(void*,const void*,const void*,const int*,uint32_t,
    ds4_session_progress_fn,void*,int,ds4_session_cancel_fn,void*,bool,bool);
#define DS41_PREFILL_BODY_RUN fixture_body_dispatch
#include "scr_engine_harness.h"
typedef struct {ds4_engine e;ds4_session s;int source[1400];} control_fixture;
static void setup_control(control_fixture *f){
    memset(f,0,sizeof(*f));f->e.backend=DS4_BACKEND_METAL;f->s=(ds4_session){.engine=&f->e,.ctx_size=8192,.ds41_graph_ready=true,.progress=progress,.display_progress=progress,.cancel=cancel};session=&f->s;CHECK(!ds4_session_set_scr_planner_budget(&f->s,5000u));
    ds41_gpu_graph *g=&f->s.ds41_graph;g->ctx=8192;g->tp_world=1;g->prefill_cap=512;g->streaming=false;g->valid=true;g->table[0].fd=g->table[1].fd=-1;
    g->token_map=malloc(DS4_N_VOCAB*sizeof(uint32_t));f->s.logits=malloc(DS4_N_VOCAB*sizeof(float));f->s.checkpoint.v=malloc(8192*sizeof(int));f->s.checkpoint.cap=8192;CHECK(g->token_map&&f->s.logits&&f->s.checkpoint.v);
    for(uint32_t i=0;i<DS4_N_VOCAB;++i)g->token_map[i]=i^73u;
    uint64_t live=0;for(uint32_t o=0;o<4;++o){uint32_t ratio=o<3?2:1;g->compressed[o]=ds4_gpu_tensor_alloc(8192ull/ratio*512*4);g->index_cache[o]=ds4_gpu_tensor_alloc(8192ull/ratio*128*4);CHECK(g->compressed[o]&&g->index_cache[o]);live+=8192ull/ratio*640*4;}
    for(uint32_t l=0;l<40;++l){g->window[l]=ds4_gpu_tensor_alloc(sizeof(values));CHECK(g->window[l]);live+=sizeof(values);}g->allocation_bytes=f->e.ds41_session_bytes=live;
    rows=ds4_gpu_tensor_alloc(sizeof(values));indices=ds4_gpu_tensor_alloc(DS41_RAW_ARENA_TOKENS*128u*4u);CHECK(rows&&indices);
    f->e.vocab.n_vocab=DS4_N_VOCAB;f->e.vocab.token=calloc(DS4_N_VOCAB,sizeof(ds4_str));CHECK(f->e.vocab.token);for(uint32_t i=0;i<DS4_N_VOCAB;++i)f->e.vocab.token[i]=(ds4_str){.ptr="\xc4\x8a",.len=2};f->e.vocab.token[277]=(ds4_str){.ptr="\xc4\xa0",.len=2};table_init(&f->e.vocab.token_to_id,1);table_put(&f->e.vocab.token_to_id,f->e.vocab.token[277],277);
    char err[256];CHECK(!ds4_session_scr_set_padding_policy(&f->s,DS4_SCR_PADDING_PLANNER_FRESH_GAP_SPACE,err,sizeof(err)));
    for(uint32_t i=0;i<1400;++i)f->source[i]=40000+(int)i;
}
static void cleanup_control(control_fixture *f){
    controls(f->source);CHECK(!ds4_session_scr_archive_release(&f->s));ds41_session_coordinates_clear(&f->s);ds41_gpu_graph *g=&f->s.ds41_graph;
    ds4_gpu_tensor_free(rows);ds4_gpu_tensor_free(indices);for(uint32_t o=0;o<4;++o){ds4_gpu_tensor_free(g->compressed[o]);ds4_gpu_tensor_free(g->index_cache[o]);}for(uint32_t l=0;l<40;++l)ds4_gpu_tensor_free(g->window[l]);free(g->token_map);free(f->s.logits);free(f->s.checkpoint.v);table_free(&f->e.vocab.token_to_id);free(f->e.vocab.token);session=NULL;
}
static bool wrapper_only, wrapper_result, expected_only, expected_resume;
static unsigned wrapper_calls;
static void *expected_g, *expected_m, *expected_w, *expected_ud;
static const int *expected_tokens;
static bool fixture_body_dispatch(void*g,const void*m,const void*w,const int*v,uint32_t n,
    ds4_session_progress_fn p,void*ud,int total,ds4_session_cancel_fn c,void*cu,bool only,bool resume){
    if(wrapper_only){CHECK(g==expected_g&&m==expected_m&&w==expected_w&&v==expected_tokens&&n==17&&total==91&&ud==expected_ud&&cu==expected_ud&&p==progress&&c==cancel&&only==expected_only&&resume==expected_resume);++wrapper_calls;return wrapper_result;}
    CHECK(session&&session->scr_capture_batch_chunks&&!only&&!resume);
    return model_prefill(g,m,w,v,n,p,ud,total,c,cu);
}
static void per_call_cases(void){
    int v[17]={0},tag;expected_g=&tag;expected_m=&tag;expected_w=&tag;expected_ud=&tag;expected_tokens=v;wrapper_only=true;
    for(unsigned i=0;i<2;i++){wrapper_result=i;
        CHECK(ds41_graph_prefill(expected_g,expected_m,expected_w,v,17,progress,&tag,91,cancel,&tag)==wrapper_result);
        CHECK(ds41_graph_prefill_sweep(expected_g,expected_m,expected_w,v,17,progress,&tag,91,cancel,&tag,false,false)==wrapper_result);
        expected_only=true;CHECK(ds41_graph_prefill_sweep(expected_g,expected_m,expected_w,v,17,progress,&tag,91,cancel,&tag,true,false)==wrapper_result);
        expected_only=false;expected_resume=true;CHECK(ds41_graph_prefill_sweep(expected_g,expected_m,expected_w,v,17,progress,&tag,91,cancel,&tag,false,true)==wrapper_result);expected_resume=false;}
    CHECK(wrapper_calls==8);wrapper_only=false;
}
static unsigned fault, chunk_attempts, chunk_success, chunk_rows, scalar_calls, dispatched_rows;
static unsigned partial_chunks, full_chunks, work_after_fault, max_installs;
static bool fault_seen;
static ds41_scr_archive *pinned_source;
static uint64_t pinned_generation;
static bool changed_rows(ds41_gpu_graph*g,uint32_t n){
    const ds41_policy_owner*o=session?session->scr_dense_owner:NULL;
    return o&&o->capture_batch_chunks&&session->scr_capture_batch_chunks&&g->archive_capture_pending&&installs&&n>1;
}
static bool chunk_before_rows(void*p,uint32_t n){
    ds41_gpu_graph*g=p;const ds41_policy_owner*o=session?session->scr_dense_owner:NULL;
    if(o)target=o->full->v;
    if(fault_seen)++work_after_fault;
    if(installs>max_installs)max_installs=installs;
    if(!changed_rows(g,n)){if(installs&&n==1)scalar_calls++;return false;}
    ++chunk_attempts;dispatched_rows+=n;CHECK(g->archive_pending&&n<=g->archive_pending->raw_arena_tokens&&g->pos>=g->archive_pending->raw_origin&&n<=g->archive_pending->raw_origin+g->archive_pending->raw_arena_tokens-g->pos);
    if(!pinned_source){pinned_source=g->archive;pinned_generation=pinned_source->generation;}
    CHECK(g->archive==pinned_source&&g->archive->generation==pinned_generation);
    if(fault==1&&!fault_seen){fault_seen=true;capture(g,g->pos,2);g->pos+=2;g->valid=false;return true;}
    return false;
}
static void chunk_after_rows(void*p,const int*v,uint32_t n){
    ds41_gpu_graph*g=p;
    for(uint32_t j=0;j<n;j++){for(uint32_t i=2;i>0;i--)g->history.tail[i]=g->history.tail[i-1];g->history.tail[0]=(int32_t)g->token_map[v[j]];}
    if(!changed_rows(g,n))return;
    ++chunk_success;chunk_rows+=n;
    ds41_scr_archive*a=g->archive_pending;
    if(g->pos==a->raw_origin+a->raw_arena_tokens)full_chunks++;else partial_chunks++;
    CHECK(ds41_capture_chunk_tags(a,g->pos-n,g->pos));
    if(!fault_seen&&fault){fault_seen=true;
        if(fault==2)a->raw_tags[7][g->pos-1-a->raw_origin]=UINT32_MAX;
        if(fault==3)a->tags[ds41_archive_tag(a->cap,3)+g->pos-1]=UINT32_MAX;
        if(fault==4)g->history.tail[0]^=1;
        if(fault==5)g->archive->generation++;
        if(fault==6)cancel_now=true;
        if(fault==7){g->pos--;g->valid=false;}
        if(fault==8)a->rebase_predecessor=NULL;
        if(fault==9)a->tokens[g->pos-1]^=1;
    }
}
static void reset_chunk_observation(unsigned failure){fault=failure;fault_seen=false;chunk_attempts=chunk_success=chunk_rows=scalar_calls=dispatched_rows=partial_chunks=full_chunks=work_after_fault=max_installs=0;pinned_source=NULL;pinned_generation=0;}
static int logical_call(control_fixture*f,int*logical,uint32_t n,uint32_t k,ds4_scr_logical_receipt*r){
    char err[256];ds4_tokens input={logical,(int)n,(int)n};controls(logical);
    int rc=ds4_session_scr_logical_policy(&f->s,&input,4096,k,201,r,err,sizeof(err));
    if(rc)fprintf(stderr,"checked capture case fault%u rc%d: %s\n",fault,rc,err);return rc;
}
static void startup_cases(void){
    char err[128];control_fixture f;setup_control(&f);
    CHECK(!ds4_session_scr_enable_capture_batch_chunks(&f.s,err,sizeof(err)));
    CHECK(ds4_session_scr_enable_capture_batch_chunks(&f.s,err,sizeof(err))==1);
    cleanup_control(&f);
}
static void run_case(bool selected,uint32_t k,unsigned failure,bool streaming){
    control_fixture f;setup_control(&f);char err[128];if(selected)CHECK(!ds4_session_scr_enable_capture_batch_chunks(&f.s,err,sizeof(err)));
    f.source[200]=277;ds4_scr_logical_receipt r;
    reset_chunk_observation(0);CHECK(!logical_call(&f,f.source,1033,6,&r));CHECK(r.execution.route==DS4_SCR_DENSE_REFRESH);
    CHECK(ds4_session_scr_enable_capture_batch_chunks(&f.s,err,sizeof(err))==1);
    CHECK(f.s.scr_capture_batch_chunks == selected);
    int logical[801],unchanged[801];for(uint32_t i=0;i<800;i++)logical[i]=80000+(int)i;
    memcpy(logical+17,f.source+128,128*sizeof(int));memcpy(logical+337,f.source+512,128*sizeof(int));memcpy(unchanged,logical,800*sizeof(int));
    f.s.ds41_graph.streaming=streaming;
    reset_chunk_observation(failure);int rc=logical_call(&f,logical,800,k,&r);
    ds4_scr_multi_receipt receipt=r.execution.multi;
    CHECK(!memcmp(logical,unchanged,800*sizeof(int))&&!work_after_fault);
    if(failure){
        CHECK(rc&&!r.published&&!r.execution.multi.published&&!f.s.checkpoint_valid&&fault_seen&&chunk_attempts==1);
        CHECK(r.execution.multi.partial_work_unknown||failure==6);
        CHECK(!r.execution.multi.gap_completed[0]||failure==6);
        CHECK(!f.s.ds41_graph.archive_pending&&!f.s.ds41_graph.archive_capture_pending);
    }else{
        CHECK(!rc&&r.published&&r.execution.multi.published&&r.execution.multi.rebase_committed&&r.planned_ranges==2&&r.useful_committed==256&&r.physical_tokens==801&&r.new_pads==1);
        CHECK(f.s.checkpoint_valid&&f.s.ds41_graph.valid&&f.s.checkpoint.len==801&&history_seen&&max_installs==2);
        int recovered[801];uint32_t recovered_n;CHECK(ds41_coordinates_project(&f.s.scr_coordinates,f.s.checkpoint.v,801,DS4_N_VOCAB,recovered,801,&recovered_n)&&recovered_n==800&&!memcmp(recovered,logical,800*sizeof(int)));
        CHECK(f.s.scr_coordinates.count==1&&f.s.checkpoint.v[f.s.scr_coordinates.pads[0]]==277&&recovered[89]==277);
        ds41_scr_archive*a=f.s.ds41_graph.archive;CHECK(a&&a->sealed&&!a->lost&&a->rebased&&a->raw&&a->raw->committed_tokens==800&&a->rebase_head_end==800);
        /* The 2048-row arena holds each gap and the suffix in one chunk. */
        if(selected&&!streaming){CHECK(chunk_attempts==2&&chunk_success==chunk_attempts&&!full_chunks&&partial_chunks==2&&chunk_rows>128&&scalar_calls==1);}
        else if(!selected)CHECK(!chunk_attempts&&scalar_calls>128);
        /* Streaming batches only from 256 rows: the 192-row gap stays scalar,
         * the 334-row suffix is one arena-bounded chunk. */
        else CHECK(chunk_attempts==1&&chunk_success==1&&chunk_rows==334&&scalar_calls==193);
        /* Exact same selected input is continuation, not another chunk policy. */
        unsigned previous_attempts=chunk_attempts;CHECK(!logical_call(&f,logical,800,k,&r));CHECK(r.execution.route==DS4_SCR_DENSE_CONTINUE&&chunk_attempts==previous_attempts);
    }
    uint32_t gap=0;for(uint32_t i=0;i<DS4_SCR_MULTI_MAX_RANGES;i++)gap+=receipt.gap_completed[i];
    printf("{\"selected\":%s,\"K\":%u,\"streaming\":%s,\"fault\":%u,\"chunk_attempts\":%u,\"dispatched_chunk_rows\":%u,\"primitive_returned_rows\":%u,\"full\":%u,\"partial\":%u,\"scalar_calls\":%u,\"known_gap_rows\":%u,\"known_suffix_rows\":%u,\"unknown\":%s,\"published\":%s,\"rebase_committed\":%s,\"pending_owner_after\":%s}\n",selected?"true":"false",k,streaming?"true":"false",failure,chunk_attempts,dispatched_rows,chunk_rows,full_chunks,partial_chunks,scalar_calls,gap,receipt.suffix_completed,receipt.partial_work_unknown?"true":"false",receipt.published?"true":"false",receipt.rebase_committed?"true":"false",f.s.ds41_graph.archive_pending?"true":"false");
    cleanup_control(&f);
}
int main(void){CHECK(ds4_gpu_init());g_ds4_shape=DS4_SHAPE_FLASH41;per_call_cases();startup_cases();run_case(false,6,0,false);run_case(true,2,0,false);run_case(true,6,0,false);run_case(true,10,0,false);run_case(true,6,0,true);run_case(true,10,0,true);for(unsigned f=1;f<=9;f++)run_case(true,6,f,false);puts("batch per-call capture mechanisms passed; synthetic rows only, not numerical or quality evidence");return 0;}
