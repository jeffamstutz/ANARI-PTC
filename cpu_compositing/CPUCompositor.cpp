// ======================================================================== //
// Copyright 2024 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "CPUCompositor.h"
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <cmath>
#include <algorithm>
#include <array>
#include <limits>

namespace ptc {

  inline std::ostream &operator<<(std::ostream &o,
                                  const math::float3 v)
  { o << "(" << v.x << "," << v.y << "," << v.z << ")"; return o; }
  
  template<typename TASK_T>
  inline void parallel_for(size_t nTasks,
                           TASK_T&& taskFunction,
                           size_t blockSize=1)
  {
    if (nTasks == 0) return;
    if (nTasks == 1)
      taskFunction(size_t(0));
    else if (blockSize==1) {
      tbb::parallel_for(size_t(0), nTasks, std::forward<TASK_T>(taskFunction));
    } else {
      const size_t numBlocks = (nTasks+blockSize-1)/blockSize;
      tbb::parallel_for((size_t)0, numBlocks, [&](size_t blockIdx){
        size_t begin = blockIdx*blockSize;
        size_t end   = std::min(begin+blockSize,size_t(nTasks));
        for (size_t i=begin;i<end;i++)
          taskFunction(size_t(i));
      });
    }
  }

  template<typename TASK_T>
  void parallel_for_blocked(size_t begin, size_t end, size_t blockSize,
                            const TASK_T& taskFunction)
  {
    const size_t numTasks = end - begin;
    const size_t numBlocks = (numTasks + blockSize - 1) / blockSize;
    parallel_for(numBlocks, [&](size_t blockID) {
      size_t block_begin = begin + blockID * blockSize;
      taskFunction(block_begin, std::min(block_begin + blockSize, end));
    });
  }
  
  int divRoundUp(int a, int b) { return (a+b-1)/b; }

  /*! depth that stands in for 'nothing hit' while sorting; any input depth
      at or beyond it (+inf, or FLT_MAX as helide writes), and NaN, is a
      miss */
  constexpr float noHitDepth = 1e20f;

  /*! one rank's fragment of a pixel, for sorting: nearest first, and on
      equal depth by rank, so the order does not depend on the sort */
  struct Fragment {
    float    depth;
    uint32_t rank;

    bool operator<(const Fragment &other) const
    {
      return depth < other.depth
        || (depth == other.depth && rank < other.rank);
    }
  };

  /*! [0,1] to 0..255, rounded; out-of-range values (and NaN) clamp */
  inline uint32_t toUnorm8(float v)
  {
    v = v * 255.f + .5f;
    if (!(v > 0.f)) return 0;
    if (v >= 255.f) return 255;
    return uint32_t(v);
  }

  /*! ColorFormat::RGBA8: one uint32_t per pixel, r in the low byte */
  struct RGBA8Pixel {
    using Element = uint32_t;
    static math::float4 load(const Element *buf, size_t i)
    {
      const uint32_t v = buf[i];
      return math::float4(float((v >>  0) & 0xff),
                          float((v >>  8) & 0xff),
                          float((v >> 16) & 0xff),
                          float((v >> 24) & 0xff)) * (1.f/255.f);
    }
    static void store(Element *buf, size_t i, const math::float4 &c)
    {
      buf[i] = toUnorm8(c.x)
        | (toUnorm8(c.y) << 8)
        | (toUnorm8(c.z) << 16)
        | (toUnorm8(c.w) << 24);
    }
  };

  /*! IEC 61966-2-1 sRGB transfer functions, as helide's (and GPU sRGB
      formats') */
  inline float srgbToLinear(float v)
  {
    return v <= 0.04045f ? v * (1.f/12.92f)
      : std::pow((v + 0.055f) * (1.f/1.055f), 2.4f);
  }
  inline float linearToSrgb(float v)
  {
    return v <= 0.0031308f ? 12.92f * v
      : 1.055f * std::pow(v, 1.f/2.4f) - 0.055f;
  }

  /*! ColorFormat::RGBA8_SRGB: as RGBA8, but rgb are sRGB-encoded (from
      premultiplied linear rgb, per channel); load() decodes to linear and
      store() encodes again. Alpha is linear. */
  struct RGBA8SRGBPixel {
    using Element = uint32_t;
    static math::float4 load(const Element *buf, size_t i)
    {
      static const auto toLinear = [] {
        std::array<float, 256> lut;
        for (int code=0;code<256;code++)
          lut[code] = srgbToLinear(code * (1.f/255.f));
        return lut;
      }();
      const uint32_t v = buf[i];
      return math::float4(toLinear[(v >>  0) & 0xff],
                          toLinear[(v >>  8) & 0xff],
                          toLinear[(v >> 16) & 0xff],
                          ((v >> 24) & 0xff) * (1.f/255.f));
    }
    static void store(Element *buf, size_t i, const math::float4 &c)
    {
      RGBA8Pixel::store(buf, i, math::float4(linearToSrgb(c.x),
                                             linearToSrgb(c.y),
                                             linearToSrgb(c.z),
                                             c.w));
    }
  };

  /*! ColorFormat::RGBA32F: four floats per pixel, r first */
  struct RGBA32FPixel {
    using Element = float;
    static math::float4 load(const Element *buf, size_t i)
    {
      const Element *p = buf + 4*i;
      return math::float4(p[0], p[1], p[2], p[3]);
    }
    static void store(Element *buf, size_t i, const math::float4 &c)
    {
      Element *p = buf + 4*i;
      p[0] = c.x;
      p[1] = c.y;
      p[2] = c.z;
      p[3] = c.w;
    }
  };

  void CPUCompositor::ColorBuffer::resize(ColorFormat format, size_t numPixels)
  {
    // only keep the buffer for the format in use
    if (format == ColorFormat::RGBA32F) {
      rgba8 = {};
      rgba32f.resize(4*numPixels);
    } else {
      rgba32f = {};
      rgba8.resize(numPixels);
    }
  }

  void *CPUCompositor::ColorBuffer::data(ColorFormat format)
  {
    return format == ColorFormat::RGBA32F
      ? (void *)rgba32f.data()
      : (void *)rgba8.data();
  }

  int CPUCompositor::colorElementsPerPixel() const
  {
    return format == ColorFormat::RGBA32F ? 4 : 1;
  }

  MPI_Datatype CPUCompositor::colorElementType() const
  {
    return format == ColorFormat::RGBA32F ? MPI_FLOAT : MPI_UINT32_T;
  }

  CPUCompositor::CPUCompositor(MPI_Comm comm)
  {
    mpi.comm = comm;
    MPI_Comm_rank(comm,&mpi.rank);
    MPI_Comm_size(comm,&mpi.size);
    printf("#cpu compositor starting up, ranks %i/%i\n",mpi.rank,mpi.size);
    p2p.sendOffset.resize(mpi.size);
    p2p.sendCount.resize(mpi.size);
    p2p.recvOffset.resize(mpi.size);
    p2p.recvCount.resize(mpi.size);
    p2p.colorSendOffset.resize(mpi.size);
    p2p.colorSendCount.resize(mpi.size);
    p2p.colorRecvOffset.resize(mpi.size);
    p2p.colorRecvCount.resize(mpi.size);

    if (mpi.rank == 0) {
      finalGather.rank0.recvCount.resize(mpi.size);
      finalGather.rank0.recvOffset.resize(mpi.size);
    }
  }

  CPUCompositor::~CPUCompositor()
  {
    // best effort: returns false without waiting if anything else (another
    // compositor, or other TBB users in the process) still holds the
    // scheduler; in that case whoever releases it last has to wait
    tbb::finalize(tbbHandle, std::nothrow);
  }

  inline std::ostream &operator<<(std::ostream &o,
                                  const CPUCompositor::Range range)
  { o << "(" << range.begin << "," << range.end << ")"; return o; }
  
  CPUCompositor::Range CPUCompositor::rangeOfPeer(int peer)
  {
    int numPixels = size.x*size.y;
    int numChunks = divRoundUp(numPixels,(int)chunkSize);
    int chunk_begin = (peer * numChunks) / mpi.size;
    int chunk_end = ((peer+1) * numChunks) / mpi.size;
    return Range(chunk_begin*chunkSize,
                 std::min(chunk_end*chunkSize,numPixels));
  }
  
  void CPUCompositor::resize(int sx, int sy, ColorFormat format)
  {
    assert(sx > 0);
    assert(sy > 0);
    size = { sx,sy };
    this->format = format;

    MPI_Barrier(mpi.comm);
    
    Range myRange = rangeOfPeer(mpi.rank);
    int recvBufSize = myRange.size() * mpi.size;
    p2p.recvBuf.color.resize(format, recvBufSize);
    p2p.recvBuf.depth.resize(recvBufSize);
    finalGather.anyRank.sendCount  = myRange.size();
    finalGather.anyRank.sendBuf.color.resize(format, myRange.size());
    finalGather.anyRank.sendBuf.depth.resize(myRange.size());

    if (mpi.rank == 0) {
      /* compute receive offsets into FINAL frame buffer (Only need
         that on rank 0, but anyway...) */
      for (int peer=0;peer<mpi.size;peer++) {
        Range peerRange = rangeOfPeer(peer);
        finalGather.rank0.recvOffset[peer] = peerRange.begin;
        finalGather.rank0.recvCount[peer]  = peerRange.size();
      }
    }
      
    { /* compute receive offsets into peer-to-peer exchange state */
      const int elems = colorElementsPerPixel();
      int recvOffset = 0;
      int sendOffset = 0;
      for (int peer=0;peer<mpi.size;peer++) {
        Range peerRange = rangeOfPeer(peer);

        p2p.sendOffset[peer] = sendOffset;
        p2p.sendCount[peer] = peerRange.size();
        
        p2p.recvOffset[peer] = recvOffset;
        p2p.recvCount[peer] = myRange.size();

        p2p.colorSendOffset[peer] = elems * p2p.sendOffset[peer];
        p2p.colorSendCount[peer]  = elems * p2p.sendCount[peer];
        p2p.colorRecvOffset[peer] = elems * p2p.recvOffset[peer];
        p2p.colorRecvCount[peer]  = elems * p2p.recvCount[peer];

        recvOffset += p2p.recvCount[peer];
        sendOffset += p2p.sendCount[peer];
      }
    }
    MPI_Barrier(mpi.comm);
  }

  /*! blends each of my pixels' fragments (one per rank, from the p2p
      receive buffers) into the final gather's send buffers */
  template <typename PIXEL_T>
  void CPUCompositor::compositeMyRange()
  {
    using Element = typename PIXEL_T::Element;
    const Element *colors
      = (const Element *)p2p.recvBuf.color.data(format);
    Element *outColors
      = (Element *)finalGather.anyRank.sendBuf.color.data(format);
    const float *depths = p2p.recvBuf.depth.data();
    float *outDepths = finalGather.anyRank.sendBuf.depth.data();

    const size_t numPixels = (size_t)rangeOfPeer(mpi.rank).size();
    // rank r's fragment of pixel i is at [i + r*numPixels]
    const size_t rankStep = numPixels;
    parallel_for_blocked
      (0,numPixels,blockSize,[&](size_t begin,size_t end) {
        std::vector<Fragment> fragments(mpi.size);
        for (size_t it=begin;it<end;it++) {
          for (size_t r=0;r<fragments.size();r++) {
            float depth = depths[it + r*rankStep];
            if (!(depth < noHitDepth)) depth = noHitDepth;
            fragments[r] = { depth, (uint32_t)r };
          }
          std::sort(fragments.begin(), fragments.end());

          // back to front: each fragment goes over what is behind it;
          // colors are premultiplied, so rgb and alpha blend alike
          math::float4 composited
            = PIXEL_T::load(colors, it + fragments.back().rank*rankStep);
          for (size_t i=fragments.size()-1;i-- > 0;) {
            const math::float4 frag
              = PIXEL_T::load(colors, it + fragments[i].rank*rankStep);
            composited = frag + (1.f-frag.w) * composited;
          }

          float final_depth = fragments.front().depth;
          if (final_depth >= noHitDepth)
            final_depth = std::numeric_limits<float>::infinity();

          PIXEL_T::store(outColors, it, composited);
          outDepths[it] = final_depth;
        }
      });
  }
  
  /*! out_color and out_depth should be null on all ranks > 0 */
  void CPUCompositor::run(void        *out_color,
                          float       *out_depth,
                          const void  *in_color,
                          const float *in_depth)
  {
    // ------------------------------------------------------------------
    // peer-to-peer stage
    // ------------------------------------------------------------------
    MPI_Alltoallv(in_color,
                  p2p.colorSendCount.data(),
                  p2p.colorSendOffset.data(),
                  colorElementType(),
                  p2p.recvBuf.color.data(format),
                  p2p.colorRecvCount.data(),
                  p2p.colorRecvOffset.data(),
                  colorElementType(),
                  mpi.comm);
    MPI_Alltoallv(in_depth,
                  p2p.sendCount.data(),
                  p2p.sendOffset.data(),
                  MPI_FLOAT,
                  p2p.recvBuf.depth.data(),
                  p2p.recvCount.data(),
                  p2p.recvOffset.data(),
                  MPI_FLOAT,
                  mpi.comm);

    // ------------------------------------------------------------------
    // local compositing stage
    // ------------------------------------------------------------------
    switch (format) {
    case ColorFormat::RGBA32F:
      compositeMyRange<RGBA32FPixel>();
      break;
    case ColorFormat::RGBA8_SRGB:
      compositeMyRange<RGBA8SRGBPixel>();
      break;
    default:
      compositeMyRange<RGBA8Pixel>();
    }

    // ------------------------------------------------------------------
    // final gather stage
    // ------------------------------------------------------------------
    enum { colorTag = 0, depthTag = 1 };
    const int elems = colorElementsPerPixel();
    // both element types (uint32_t, float) are 4 bytes
    static_assert(sizeof(uint32_t) == 4 && sizeof(float) == 4, "");
    const size_t colorBytesPerPixel = size_t(elems) * 4;
    std::vector<MPI_Request> requests;
    MPI_Request request;
    if (mpi.rank == 0) {
      for (int r=0;r<mpi.size;r++) {
        const int offset = finalGather.rank0.recvOffset[r];
        const int count  = finalGather.rank0.recvCount[r];
        MPI_Irecv((char *)out_color + size_t(offset)*colorBytesPerPixel,
                  elems*count,
                  colorElementType(),r,colorTag,mpi.comm,&request);
        requests.push_back(request);
        MPI_Irecv(out_depth+offset,
                  count,
                  MPI_FLOAT,r,depthTag,mpi.comm,&request);
        requests.push_back(request);
      }
    }
    /*everybody:*/{
      MPI_Isend(finalGather.anyRank.sendBuf.color.data(format),
                elems*finalGather.anyRank.sendCount,
                colorElementType(),0,colorTag,mpi.comm,&request);
      requests.push_back(request);
      MPI_Isend(finalGather.anyRank.sendBuf.depth.data(),
                finalGather.anyRank.sendCount,
                MPI_FLOAT,0,depthTag,mpi.comm,&request);
      requests.push_back(request);
    }

    MPI_Waitall(requests.size(),
                requests.data(),
                MPI_STATUSES_IGNORE);
    MPI_Barrier(mpi.comm);
  }
  
}
