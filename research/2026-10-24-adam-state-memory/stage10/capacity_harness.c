/* Independent external test harness: compare complete arenas with and without the rejected attempt. */
#define main factor_replay_cli_main
#include "factor_replay.c"
#undef main
int main(void){
  uint32_t seeds[]={0,42}, caps[]={0,1,2,4,8,15};
  for(size_t si=0;si<2;si++)for(uint32_t mode=1;mode<=2;mode++)for(size_t ci=0;ci<6;ci++){
    uint32_t cap=caps[ci],block=128;size_t hist=sizeof(Record)*cap;
    size_t work=4*(2*block+(mode==1?PARAMS:block));size_t req=sizeof(Run)+4*PARAMS+hist+work;
    Run*a=calloc(1,req),*b=calloc(1,req);if(!a||!b)return 3;
    a->mode=b->mode=mode;a->seed=b->seed=seeds[si];a->steps=cap;b->steps=cap+1;
    a->block=b->block=block;a->capacity=b->capacity=cap;algorithm(a,NULL);algorithm(b,NULL);
    int we=!memcmp(weights(a),weights(b),4*PARAMS),he=!memcmp(records(a),records(b),hist),se=!memcmp(workspace(a),workspace(b),work);
    int ce=a->capture_calls==b->capture_calls&&a->rebuild_calls==b->rebuild_calls&&a->reconstructed_coordinates==b->reconstructed_coordinates&&a->gradient_products==b->gradient_products&&a->moment_transitions==b->moment_transitions;
    int de=a->gradient_digest==b->gradient_digest&&a->state_digest==b->state_digest&&a->factor_digest==b->factor_digest;
    int ok=we&&he&&se&&ce&&de&&a->completed==cap&&b->completed==cap&&a->status==0&&b->status==1&&b->rejected_step==cap+1;
    printf("{\"mode\":%u,\"seed\":%u,\"capacity\":%u,\"weight_bytes_equal\":%s,\"history_bytes_equal\":%s,\"workspace_bytes_equal\":%s,\"counters_equal\":%s,\"digests_equal\":%s,\"all_passed\":%s}\n",mode,seeds[si],cap,we?"true":"false",he?"true":"false",se?"true":"false",ce?"true":"false",de?"true":"false",ok?"true":"false");
    free(a);free(b);if(!ok)return 4;
  }
  return 0;
}
