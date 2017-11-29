#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <signal.h>
#include <pthread.h>
#include <semaphore.h>

#include "srslte/phy/common/phy_common.h"
#include "srslte/phy/io/filesink.h"
#include "srslte/phy/rf/rf.h"
#include "srslte/phy/rf/rf_utils.h"
#include "srslte/srslte.h"
#include "srsgui/srsgui.h"

#define ID 128
#define SAMPLING_RATE 10
#define PHASE_WRAP(x) (x - 2 * M_PI * floorf((x + M_PI) / (2 * M_PI)))
// #define USE_PLOT

void init_plots();
void* plot_thread_run(void*);

typedef struct {
    char*    rf_args;
    uint32_t rf_nof_rx_ant;
    float    rf_gain;
    double   rf_freq;
    
    char*    net_address; 
    int      net_port; 

    int      nof_subframes;
    int      cpu_affinity;
    int      force_N_id_2;
    bool     disable_cfo;
    uint16_t rnti;
    int      verbose;
} prog_args_t;

void args_default(prog_args_t *args) {
    args->rf_args = "";
    args->rf_nof_rx_ant = 2;
    args->rf_gain = 31.5;
    args->rf_freq = -1.0;
    
    args->net_address = "127.0.0.1";
    args->net_port = -1; 

    args->nof_subframes = -1;
    args->cpu_affinity = -1;
    args->force_N_id_2 = -1;
    args->disable_cfo = true;
    args->rnti = SRSLTE_SIRNTI;
}

void usage(prog_args_t *args, char *prog) {
    printf("Usage: %s [agnv] -f rx_frequency (in Hz)\n", prog);
    printf("\t-a Baseband RF args [Default %s]\n", args->rf_args);
    printf("\t-n nof_subframes [Default %d]\n", args->nof_subframes);
    printf("\t-h remote TCP address to send data [Default %s]\n", args->net_address);
    printf("\t-p remote TCP port to send data (-1 does nothing with it) [Default %d]\n", args->net_port);
    printf("\t-v [set srslte_verbose to debug, default none]\n");
}

void parse_args(prog_args_t *args, int argc, char **argv) {
    int opt;
    args_default(args);
    while ((opt = getopt(argc, argv, "anhpvf")) != -1) {
        switch (opt) {
            case 'a':
                args->rf_args = argv[optind];
                break;
            case 'n':
                args->nof_subframes = atoi(argv[optind]);
                break;
            case 'h':
                args->net_address = argv[optind];
                break;
            case 'p':
                args->net_port = atoi(argv[optind]);
                break;
            case 'v':
                srslte_verbose++;
                args->verbose = srslte_verbose;
                break;
            case 'f':
                args->rf_freq = strtod(argv[optind], NULL);
                break;
            default:
                usage(args, argv[0]);
                exit(-1);
        }
    }
    if (args->rf_freq < 0) {
        usage(args, argv[0]);
        exit(-1);
    }
}

prog_args_t prog_args;

cell_search_cfg_t cell_detect_config = {
    SRSLTE_DEFAULT_MAX_FRAMES_PBCH,
    SRSLTE_DEFAULT_MAX_FRAMES_PSS,
    SRSLTE_DEFAULT_NOF_VALID_PSS_FRAMES,
    0
};

srslte_rf_t rf;
srslte_cell_t cell;
srslte_ue_sync_t ue_sync;
srslte_ue_mib_t ue_mib;
srslte_ue_dl_t ue_dl;
srslte_netsink_t net_sink;

pthread_t plot_thread;
sem_t plot_sem;

char input[128];
uint8_t *data[SRSLTE_MAX_CODEWORDS];
cf_t *sf_buffer[SRSLTE_MAX_PORTS] = {NULL};
uint8_t bch_payload[SRSLTE_BCH_PAYLOAD_LEN];

uint8_t* packet;
uint8_t packet_size;

uint32_t sf_cnt = 0;
uint32_t sfn = 0;
int sfn_offset;

float cfo = 0;

enum receiver_state { DECODE_MIB, DECODE_PDSCH } state;

bool go_exit = false;
void sig_int_handler(int signo)
{
    printf("SIGINT received. Exiting...\n");
    if (signo == SIGINT) {
        go_exit = true;
    } else if (signo == SIGSEGV) {
        exit(1);
    }
}

int srslte_rf_recv_wrapper(void *h, cf_t *data[SRSLTE_MAX_PORTS], uint32_t nsamples, srslte_timestamp_t *t) {
    DEBUG(" ----  Receive %d samples  ---- \n", nsamples);
    void *ptr[SRSLTE_MAX_PORTS];
    for (int i=0;i<SRSLTE_MAX_PORTS;i++) {
        ptr[i] = data[i];
    }
    return srslte_rf_recv_with_time_multi(h, ptr, nsamples, true, NULL, NULL);
}

int main(int argc, char **argv) {
    parse_args(&prog_args, argc, argv);

    for (int i = 0; i < SRSLTE_MAX_CODEWORDS; i++) {
        data[i] = srslte_vec_malloc(sizeof(uint8_t)*1500*8);
        if (!data[i]) {
            ERROR("Allocating data");
            go_exit = true;
        }
    }
    
    if (prog_args.net_port > 0) {
        if (srslte_netsink_init(&net_sink, prog_args.net_address, prog_args.net_port, SRSLTE_NETSINK_TCP)) {
            fprintf(stderr, "Error initiating TCP socket to %s:%d\n", prog_args.net_address, prog_args.net_port);
            exit(-1);
        }
        srslte_netsink_set_nonblocking(&net_sink);
    }

    if(prog_args.cpu_affinity > -1) {
        cpu_set_t cpuset;
        pthread_t thread;

        thread = pthread_self();
        for(int i = 0; i < 8;i++){
            if(((prog_args.cpu_affinity >> i) & 0x01) == 1){
                printf("Setting pdsch_ue with affinity to core %d\n", i);
                CPU_SET((size_t) i , &cpuset);
            }
            if(pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)){
                fprintf(stderr, "Error setting main thread affinity to %d \n", prog_args.cpu_affinity);
                exit(-1);
            }
        }
    }
    
    printf("Opening baseband RF device with %d RX antennas...\n", prog_args.rf_nof_rx_ant);
    if (srslte_rf_open_multi(&rf, prog_args.rf_args, prog_args.rf_nof_rx_ant)) {
        fprintf(stderr, "Error opening rf\n");
        exit(-1);
    }
    srslte_rf_set_rx_gain(&rf, prog_args.rf_gain);

    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGINT);
    sigprocmask(SIG_UNBLOCK, &sigset, NULL);
    signal(SIGINT, sig_int_handler);

    srslte_rf_set_master_clock_rate(&rf, 30.72e6);

    printf("Tunning receiver to %.3f MHz\n", prog_args.rf_freq / 1000000);
    srslte_rf_set_rx_freq(&rf, prog_args.rf_freq);
    srslte_rf_rx_wait_lo_locked(&rf);

    
    int ret;
    uint32_t ntrial = 0;
    do {
        ret = rf_search_and_decode_mib(&rf, prog_args.rf_nof_rx_ant, &cell_detect_config, prog_args.force_N_id_2, &cell, &cfo);
        if (ret < 0) {
            fprintf(stderr, "Error searching for cell\n");
            exit(-1);
        } else if (ret == 0 && !go_exit) {
            printf("Cell not found after %d trials. Trying again (Press Ctrl+C to exit)\n", ntrial++);
        }
    } while (ret == 0 && !go_exit);

    if (go_exit) {
        srslte_rf_close(&rf);
        exit(0);
    }

    srslte_rf_stop_rx_stream(&rf);
    srslte_rf_flush_buffer(&rf);

    int srate = srslte_sampling_freq_hz(cell.nof_prb);
    if (srate != -1) {
        if (srate < 10e6) {
            srslte_rf_set_master_clock_rate(&rf, 4*srate);
        } else {
            srslte_rf_set_master_clock_rate(&rf, srate);
        }
        printf("Setting sampling rate %.2f MHz\n", (float) srate/1000000);
        float srate_rf = srslte_rf_set_rx_srate(&rf, (double) srate);
        if (srate_rf != srate) {
            fprintf(stderr, "Could not set sampling rate\n");
            exit(-1);
        }
    } else {
        fprintf(stderr, "Invalid number of PRB %d\n", cell.nof_prb);
        exit(-1);
    }

    if (srslte_ue_sync_init_multi_decim(
            &ue_sync,
            cell.nof_prb,
            cell.id==1000,
            srslte_rf_recv_wrapper,
            prog_args.rf_nof_rx_ant,
            (void*) &rf, 0)) {
        fprintf(stderr, "Error initiating ue_sync\n");
        exit(-1);
    }
    if (srslte_ue_sync_set_cell(&ue_sync, cell)) {
        fprintf(stderr, "Error initiating ue_sync\n");
        exit(-1);
    }
    ue_sync.correct_cfo = !prog_args.disable_cfo;
    srslte_ue_sync_set_cfo(&ue_sync, cfo);


    if (srslte_ue_mib_init(&ue_mib, cell.nof_prb)) {
        fprintf(stderr, "Error initaiting UE MIB decoder\n");
        exit(-1);
    }
    if (srslte_ue_mib_set_cell(&ue_mib, cell)) {
        fprintf(stderr, "Error initaiting UE MIB decoder\n");
        exit(-1);
    }
    srslte_pbch_decode_reset(&ue_mib.pbch);


    if (srslte_ue_dl_init(&ue_dl, cell.nof_prb, prog_args.rf_nof_rx_ant)) {
        fprintf(stderr, "Error initiating UE downlink processing module\n");
        exit(-1);
    }
    if (srslte_ue_dl_set_cell(&ue_dl, cell)) {
        fprintf(stderr, "Error initiating UE downlink processing module\n");
        exit(-1);
    }
    srslte_ue_dl_set_rnti(&ue_dl, prog_args.rnti);

    for (int i = 0; i < prog_args.rf_nof_rx_ant; i++) {
        sf_buffer[i] = srslte_vec_malloc(3 * sizeof(cf_t) * SRSLTE_SF_LEN_PRB(cell.nof_prb));
    }

    if (prog_args.net_port > 0) {
        packet_size = 5 + ue_dl.cell.nof_prb * (sizeof(float) / sizeof(uint8_t));
        packet = srslte_vec_malloc(packet_size * sizeof(uint8_t));
        packet[0] = '$';
        packet[1] = ID;
        packet[2] = 0;
        packet[packet_size - 2] = '\r';
        packet[packet_size - 1] = '\n';

        uint8_t header[] = {
            '$', 
            ID, 
            (uint8_t)packet_size, 
            '\r', 
            '\n'
        };
        srslte_netsink_write(&net_sink, NULL, 0);
        srslte_netsink_write(&net_sink, header, sizeof(header));
    }

    init_plots();
    srslte_rf_start_rx_stream(&rf);

    INFO("\nEntering main loop...\n\n", 0);
    while (!go_exit && (sf_cnt < prog_args.nof_subframes || prog_args.nof_subframes == -1)) {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(0, &set);

        struct timeval to = (struct timeval) {
            .tv_sec = 0,
            .tv_usec = 0
        };

        srslte_verbose = prog_args.verbose;
        int n = select(1, &set, NULL, NULL, &to);
        if (n == 1) {
            if (fgets(input, sizeof(input), stdin)) {
                printf(input);
            }
        }

        ret = srslte_ue_sync_zerocopy_multi(&ue_sync, sf_buffer);
        if (ret < 0) {
            fprintf(stderr, "Error calling srslte_ue_sync_work()\n");
        }

        float sample_offset = (float) srslte_ue_sync_get_last_sample_offset(&ue_sync) + srslte_ue_sync_get_sfo(&ue_sync) / 1000;
        srslte_ue_dl_set_sample_offset(&ue_dl, sample_offset);

        if (ret == 1) {
            uint32_t sfidx = srslte_ue_sync_get_sfidx(&ue_sync);

            switch (state) {
                case DECODE_MIB:
                    if (sfidx == 0) {
                        int ret_mib = srslte_ue_mib_decode(&ue_mib, sf_buffer[0], bch_payload, NULL, &sfn_offset);
                        if (ret_mib < 0) {
                            fprintf(stderr, "Error decoding UE MIB\n");
                            exit(-1);
                        } else if (ret_mib == SRSLTE_UE_MIB_FOUND) {
                            srslte_pbch_mib_unpack(bch_payload, &cell, &sfn);
                            srslte_cell_fprint(stdout, &cell, sfn);
                            printf("Decoded MIB. SFN: %d, offset: %d\n", sfn, sfn_offset);
                            sfn = (sfn + sfn_offset) % 1024;
                            state = DECODE_PDSCH;
                        }
                    }
                    break;

                case DECODE_PDSCH:
                    for (int j = 0; j < ue_dl.nof_rx_antennas; j++) {
                        srslte_ofdm_rx_sf(&ue_dl.fft, sf_buffer[j], ue_dl.sf_symbols_m[j]);
                    }
                    srslte_chest_dl_estimate_multi(&ue_dl.chest, ue_dl.sf_symbols_m, ue_dl.ce_m, sfidx, ue_dl.nof_rx_antennas);

                    if (prog_args.net_port > 0 && sf_cnt % (1000 / SAMPLING_RATE) == 0) {
                        packet[2] = (uint8_t) (sf_cnt % 256);

                        float* packet_data = (float*) (packet + 3);
                        for (int k = 0; k < ue_dl.cell.nof_prb; k++) {
                            packet_data[k] = PHASE_WRAP(
                                cargf(ue_dl.ce_m[0][0][k * SRSLTE_NRE]) - 
                                cargf(ue_dl.ce_m[0][1][k * SRSLTE_NRE])
                            );
                        }
                        srslte_netsink_write(&net_sink, packet, packet_size);
                    }
                    break;
            }

            if (sfidx == 9) {
                sfn = (sfn + 1) % 1024;
            }

        } else if (ret == 0) {
            printf("Finding PSS... Peak: %8.1f, FrameCnt: %d, State: %d\r",
                    srslte_sync_get_peak_value(&ue_sync.sfind),
                    ue_sync.frame_total_cnt, ue_sync.state);

            sem_post(&plot_sem);
        }

        if (sf_cnt % 100 == 0) {
            sem_post(&plot_sem);
        }

        sf_cnt++;
    } // Main loop

#ifdef USE_PLOT
    if (!pthread_kill(plot_thread, 0)) {
        pthread_kill(plot_thread, SIGHUP);
        pthread_join(plot_thread, NULL);
    }
#endif
    if (prog_args.net_port > 0) {
        free(packet);
    }
    srslte_ue_dl_free(&ue_dl);
    srslte_ue_sync_free(&ue_sync);
    for (int i = 0; i < SRSLTE_MAX_CODEWORDS; i++) {
        if (data[i]) {
            free(data[i]);
        }
    }
    for (int i = 0; i < prog_args.rf_nof_rx_ant; i++) {
        if (sf_buffer[i]) {
            free(sf_buffer[i]);
        }
    }

    srslte_ue_mib_free(&ue_mib);
    srslte_rf_close(&rf);

    printf("\nBye\n");
    exit(0);
}

/**********************************************************************
 *  Plotting Functions
 ***********************************************************************/

void init_plots() {
    if (sem_init(&plot_sem, 0, 0)) {
        perror("sem_init");
        exit(-1);
    }

#ifdef USE_PLOT
    struct sched_param param = (struct sched_param) {
        .sched_priority = 0
    };
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setschedpolicy(&attr, SCHED_OTHER);
    pthread_attr_setschedparam(&attr, &param);
    if (pthread_create(&plot_thread, NULL, plot_thread_run, NULL)) {
        perror("pthread_create");
        exit(-1);
    }
#endif
}

void *plot_thread_run(void *arg) {
    int sz = srslte_symbol_sz(ue_dl.cell.nof_prb);
    int g = (sz - SRSLTE_NRE * ue_dl.cell.nof_prb) / 2;

    plot_real_t plot[3][ue_dl.cell.nof_ports][ue_dl.nof_rx_antennas];
    float data[3][ue_dl.cell.nof_ports][ue_dl.nof_rx_antennas][sz];

    char title[3][128] = {
        "Manitude",
        "Phase",
        "Phase Difference"
    };

    float y_axis_scale[3][2] = {
        {-40.0, 40.0},
        {-M_PI, M_PI},
        {-M_PI * 1.1, M_PI * 1.1}
    };

    sdrgui_init_title("LTE");
    for (int h = 0; h < 3; h++) {
        for (int i = 0; i < ue_dl.cell.nof_ports; i++) {
            for (int j = 0; j < ue_dl.nof_rx_antennas; j++) {
                plot_real_init(&plot[h][i][j]);
                plot_real_setTitle(&plot[h][i][j], title[h]);
                plot_real_setYAxisScale(&plot[h][i][j], y_axis_scale[h][0], y_axis_scale[h][1]);
                plot_real_addToWindowGrid(&plot[h][i][j], (char*) "pdsch_ue", h, ue_dl.nof_rx_antennas * i + j);
            }
        }
    }

    while(1) {
        sem_wait(&plot_sem);

        for (int i = 0; i < ue_dl.cell.nof_ports; i++) {
            for (int j = 0; j < ue_dl.nof_rx_antennas; j++) {
                bzero(data[0][i][j], sizeof(float) * sz);
                bzero(data[1][i][j], sizeof(float) * sz);

                for (int k = 0; k < SRSLTE_NRE * ue_dl.cell.nof_prb; k++) {
                    data[0][i][j][g + k] = 20 * fmax(-4, log10f(cabsf(ue_dl.ce_m[i][j][k])));
                    data[1][i][j][g + k] = PHASE_WRAP(cargf(ue_dl.ce_m[i][j][k]));
                }

                plot_real_setNewData(&plot[0][i][j], data[0][i][j], sz);
                plot_real_setNewData(&plot[1][i][j], data[1][i][j], sz);
            }

            bzero(data[2][i][0], sizeof(float) * sz);
            for (int k = 0; k < SRSLTE_NRE * ue_dl.cell.nof_prb; k++) {
                data[2][i][0][g + k] = PHASE_WRAP(cargf(ue_dl.ce_m[i][0][k]) - cargf(ue_dl.ce_m[i][1][k]));
            }
            plot_real_setNewData(&plot[2][i][0], data[2][i][0], sz);
        }
    }

    return NULL;
}

