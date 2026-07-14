/* Standalone harness wrapping the EXACT upstream Asteroids@Home validator
   logic (period_search_validator4.cpp: init_result parse loop + compare_results),
   so we test results with the project's real acceptance criteria, not a guess.
   Only the BOINC plumbing (get_output_file_path/try_fopen/RESULT) is stubbed;
   the parse loop and comparison are byte-for-byte the project's. */
#include <cstdio>
#include <cstdlib>
#include <cmath>
using std::isnan;
#define MAX_LINES 100000
#define ERR_XML_PARSE -1

struct DATA { int nlines; double *per; double *rms; double *chisq; };

/* --- verbatim parse loop from init_result, fed a plain file path --- */
int init_from_file(const char* path, void*& data) {
    FILE* f = fopen(path, "r");
    if (!f) return -2;
    int n, nlines, max_nlines;
    double per, rms, chisq, dark, lambda, beta;
    DATA* dp = new DATA;
    max_nlines = MAX_LINES;
    dp->per=(double*)malloc(max_nlines*sizeof(double));
    dp->rms=(double*)malloc(max_nlines*sizeof(double));
    dp->chisq=(double*)malloc(max_nlines*sizeof(double));
    nlines = 0;
    while (feof(f) == 0) {
        if (nlines>=max_nlines) {
          max_nlines+=MAX_LINES;
          dp->per=(double*)realloc(dp->per,max_nlines*sizeof(double));
          dp->rms=(double*)realloc(dp->rms,max_nlines*sizeof(double));
          dp->chisq=(double*)realloc(dp->chisq,max_nlines*sizeof(double));
        }
        n = fscanf(f, "%lf %lf %lf %lf %lf %lf", &per, &rms, &chisq, &dark, &lambda, &beta);
        if (n != 6 && n != -1) { fclose(f); return ERR_XML_PARSE; }
        if (isnan(per) || isnan(rms) || isnan(chisq)) { fclose(f); return ERR_XML_PARSE; }
        dp->per[nlines] = per; dp->rms[nlines] = rms; dp->chisq[nlines] = chisq;
        nlines++;
    }
    dp->nlines = nlines;
    fclose(f);
    data = (void*) dp;
    return 0;
}

/* --- verbatim compare_results --- */
int compare_results(void* _data1, void* _data2, bool& match) {
    int i;
    double tol_per = 0.1, tol_rms = 0.1, tol_chisq = 0.5;
    DATA* data1 = (DATA*)_data1;
    DATA* data2 = (DATA*)_data2;
    match = true;
    if ((data1->nlines==0) || (data2->nlines==0) || (data1->nlines!=data2->nlines)) {
      match = false; return 0;
    }
    for (i = 0; i < data1->nlines; i++) {
        if (fabs((data1->per[i] - data2->per[i]) / (data1->per[i] + data2->per[i])) / 2 > tol_per) { match=false; break; }
        if (fabs((data1->rms[i] - data2->rms[i]) / (data1->rms[i] + data2->rms[i])) / 2 > tol_rms) { match=false; break; }
        if (fabs((data1->chisq[i] - data2->chisq[i]) / (data1->chisq[i] + data2->chisq[i])) / 2 > tol_chisq) { match=false; break; }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <result_file> <reference_file>\n", argv[0]); return 2; }
    void *d1=0, *d2=0;
    int r1 = init_from_file(argv[1], d1);
    int r2 = init_from_file(argv[2], d2);
    if (r1) { printf("PARSE-FAIL(%s) rc=%d\n", argv[1], r1); return 3; }
    if (r2) { printf("PARSE-FAIL(%s) rc=%d\n", argv[2], r2); return 3; }
    bool match=false;
    compare_results(d1, d2, match);
    DATA*a=(DATA*)d1;DATA*b=(DATA*)d2;
    /* also report the worst-line margin for insight (not part of the verdict) */
    double wp=0,wr=0,wc=0;
    if(a->nlines==b->nlines){
      for(int i=0;i<a->nlines;i++){
        double mp=fabs((a->per[i]-b->per[i])/(a->per[i]+b->per[i]))/2;
        double mr=fabs((a->rms[i]-b->rms[i])/(a->rms[i]+b->rms[i]))/2;
        double mc=fabs((a->chisq[i]-b->chisq[i])/(a->chisq[i]+b->chisq[i]))/2;
        if(mp>wp)wp=mp; if(mr>wr)wr=mr; if(mc>wc)wc=mc;
      }
    }
    printf("%-6s nlines %d/%d  worst-margin per=%.2e/0.1 rms=%.2e/0.1 chisq=%.2e/0.5\n",
           match?"VALID":"INVALID", a->nlines, b->nlines, wp, wr, wc);
    return match?0:1;
}
