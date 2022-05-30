#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>
//#include "../ray.h"

class KernelRay;
queue& getQueue(){
    static queue qCPU(sycl::INTEL::host_selector{}, async_exception_handler);
    return qCPU;
}

sycl::event cpu_submitKernel(queue& q, sycl::buffer<Primitive,1>& buf_prim, sycl::buffer<Pixel,1>& buf_pixels,
                       int width, int height, sycl::nd_range<1> size_range, int offset,
                       float camera_x,
                       float camera_y,
                       float camera_z,
                       float viewport_x,
                       float viewport_y,
                       Primitive* prim_ptr,
                       size_t n_primitives){
    q = getQueue();
    return q.submit([&](handler &h) {

#if USE_LOCAL_MEM == 1
  auto global_primitives = buf_prim.get_access<sycl::access::mode::read>(h);
  sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local>
                    local_primitives(Rprim, h);
#else
  auto primitives = buf_prim.get_access<sycl::access::mode::read>(h);
#endif

auto pixels = buf_pixels.get_access<sycl::access::mode::discard_write>(h);
auto image_width = width;
auto image_height = height;

#if QUEUE_NDRANGE
h.parallel_for<class KernelRay>(size_range, [=](nd_item<1> ndItem) {
#endif // QUEUE_NDRANGE 0

#if USE_LOCAL_MEM == 1
    auto grp = ndItem.get_group();
      auto event = grp.async_work_group_copy(local_primitives.get_pointer(), global_primitives.get_pointer(), Nprim);
      grp.wait_for(event);

      // It seems Sycl does not support using custom structs to transfer between src and dest (async_work_group_copy)
      // So we need to send floats, calculate the number of floats, and then force the conversion here:
      Primitive* primitives = static_cast<Primitive*>(static_cast<void*>((local_primitives.get_pointer().get())));
#endif

// auto grp = grp.get_id();

// grp.parallel_for_work_item([&](sycl::h_item<1> item) {
// auto tid = item.get_local_id(0);
// auto id = item.get_global_id(0);
// const auto id = ndItem.get_global_id(0);
    const auto id = ndItem.get_global_id(0);
//     // auto bid = item.get_global_id(0) / lws;
//     if (id == 0){
//       {
//       Primitive p = primitives[0];
//   static const CONSTANT char FMT[] = "p c %f (refl %f)\n";
//   sycl::intel::experimental::printf(FMT, p.m_color[0], p.m_refl);
//       }
//       {
//       Primitive p = primitives_[0];
//   static const CONSTANT char FMT[] = "p c %f (refl %f)\n";
//   sycl::intel::experimental::printf(FMT, p.m_color[0], p.m_refl);
//       }
//     }

// const auto id = ndItem.get_global_id(0);
// const auto lid = ndItem.get_local_id(0);
    const int x = (id + offset) % image_width;
    const int y = (id + offset) / image_width;
// if (y == 0 && (x == 0 || x == 31 || x == 32 || x == 63))
// {
//   // static const CONSTANT char FMT[] = "[%d] %d %d {group %d} local range %d group range %d\n";
//   // sycl::intel::experimental::printf(FMT, id, ndItem.get_global_linear_id(), ndItem.get_local_id(0), grp.get_id(0), ndItem.get_local_range(0), ndItem.get_group_range(0));
//   static const CONSTANT char FMT[] = "%s [%d] x %d y %d size %d off %d\n";
//   sycl::intel::experimental::printf(FMT, mycpu ? "cpu" : "gpu", id, x, y, mysize, myoffset);
// }
// return;
// if (id < 10 || id > 124){
//   static const CONSTANT char FMT[] = "[%d] x %d y %d (group %d tid %d bid %d) (workgroups %d)\n";
//   sycl::intel::experimental::printf(FMT, id, x, y, group, tid, bid, workgroups);
// }
//   return;

// Primitive* primitives = prims;
// const auto id = item.get_linear_id() + offset;
// const int x = id % image_width;
// const int y = id / image_width;

    if (x >= image_width || y >= image_height)
      return;

// if (id < 10){
//   static const CONSTANT char FMT[] = "[%d] x %d y %d\n";
//   sycl::intel::experimental::printf(FMT, id, x, y);
// }

// Our viewport size can be different than the image size. This lets us calculate
// the stepping within the viewport relative to the stepping within the image.
// IE with a viewport width of 6.0f and an image width of 800, each pixel is 1/800 of 6.0f
// or 0.0075f.
    const float dx = viewport_x / image_width;      // x stepping, left -> right
    const float dy = -viewport_y / image_height;    // y stepping, top -> bottom
    const float sx = -(viewport_x / 2.0f) + x * dx; // this pixel's viewport x
    const float sy = (viewport_y / 2.0f) + y * dy;  // this pixel's viewport y

// Initializes the ray queue. OpenCL has no support for recursion, so recursive ray tracing calls
// were replaced by a queue of rays that is processed in sequence. Since the recursive calls were
// additive, this works.
    /*typedef enum raytype
    {
      ORIGIN = 0,
      REFLECTED = 1,
      REFRACTED = 2
    } ray_type;

    typedef struct
    {
      float4 origin;
      float4 direction;
      float weight;
      float depth;
      int origin_primitive;
      ray_type type;
      float r_index;
      Color transparency;
    } RayK;*/

    RayK queue[MAX_RAY_COUNT];
    int rays_in_queue = 0;
    int front_ray_ptr = 0;
    int back_ray_ptr = 0;

// float4 camera = (float4)(camera_x, camera_y, camera_z, 0);
    float4 camera{camera_x, camera_y, camera_z, 0};
// Color acc = (Color)(0, 0, 0, 0);
    Color acc{0, 0, 0, 0};

// if (x < image_width/2 && y < image_height/2){
//   uchar red = 255;
//   uchar green = x;
//   uchar blue = 50;
//   Pixel p{red, green, blue, 0};
//   int pos = y * image_width + x;
//   if (x == 0){
//     static const CONSTANT char FMT[] = "p %d [%d,%d] => (%d %d %d %d)\n";
//     sycl::intel::experimental::printf(FMT, pos, x, y, p[0], p[1], p[2], p[3]);
//   }
//   pixels[pos] = p; // Pixel{red, green, blue, 0};
// }
// We use 3x supersampling to smooth out the edges in the image. This means each pixel actually
// fires 9 initial rays, plus the recursion and refraction.
    for (int tx = -1; tx < 2; tx++)
      for (int ty = -1; ty < 2; ty++) {
// Create initial ray.
// float4 dir = NORMALIZE((float4)(sx + dx * (tx / 2.0f), sy + dy * (ty / 2.0f), 0, 0) - camera);
        float4 dir = NORMALIZE((float4{sx + dx * (tx / 2.0f), sy + dy * (ty / 2.0f), 0, 0} - camera));
        RayK r;
        r.origin = camera;
        r.direction = dir;
        r.weight = 1.0f;
        r.depth = 0;
        r.origin_primitive = -1;
        r.type = ORIGIN;
        r.r_index = 1.0f;
// r.transparency = (Color)(1, 1, 1, 0);
        r.transparency = Color{1, 1, 1, 0};

// Populate queue and start the processing loop.
        PUSH_RAY(queue, r, back_ray_ptr, rays_in_queue)

        while (rays_in_queue > 0) {
          float dist;
          RayK cur_ray;
          POP_RAY(queue, cur_ray, front_ray_ptr, rays_in_queue)
// Color ray_col = (Color)(0, 0, 0, 0);
          Color ray_col{0, 0, 0, 0};
          float4 point_intersect;
          int result;
          int prim_index = -1;
// raytrace performs the actual tracing and returns useful information
// int prim_index =
//   raytrace(&cur_ray, &ray_col, &dist, &point_intersect, &result, primitives, n_primitives);
          {
// RayK* a_ray = &cur_ray;
// Color* a_acc = &ray_col;
// float* a_dist = &dist;
// float4* point_intersect_ptr = &point_intersect;
// int* result = &result;
// Primitive* primitives = primitives;
// int n_primities = n_primitives;

// int
// raytrace(RayK* a_ray,
//          Color* a_acc,
//          float* a_dist,
//          float4* point_intersect,
//          int* result,
//          Primitive* primitives,
//          int n_primitives)
            dist = MAXFLOAT;
// a_dist = MAXFLOAT;
// int prim_index = -1;

// find nearest intersection
            for (int s = 0; s < n_primitives; s++) {
              int res;
// if (res = intersect(&primitives[s], &cur_ray, &dist)) {
// TODO: review mod: previous line to next two lines:
              Primitive pp = primitives[s];
              res = intersect(&pp, &cur_ray, &dist);
              if (res) {
                prim_index = s;
                result = res;
              }
            }
// no hit
            if (prim_index != -1) { // added
// handle hit
              if (primitives[prim_index].is_light) {
                ray_col = primitives[prim_index].m_color;
              } else {
                point_intersect = cur_ray.origin + (cur_ray.direction * (dist));
// trace lights
                for (int l = 0; l < n_primitives; l++) {
                  if (primitives[l].is_light) {
// point light source shadows
                    float shade = 1.0f;
                    float L_LEN = LENGTH(primitives[l].center - point_intersect);
                    float4 L = NORMALIZE(primitives[l].center - point_intersect);
                    if (primitives[l].type == SPHERE) {
                      RayK r;
                      r.origin = point_intersect + L * EPSILON;
                      r.direction = L;
                      int s = 0;
                      while (s < n_primitives) {
//             if (&primitives[s] != &primitives[l] && !primitives[s].is_light &&
//                 intersect(&primitives[s], &r, &L_LEN)) {
// TODO: review mod: previous 2 lines to next four lines:
                        Primitive ps = primitives[s];
                        Primitive pl = primitives[l];
                        if (&ps != &pl && !ps.is_light &&
                            intersect(&ps, &r, &L_LEN)) {
                          shade = 0;
                        }
                        s++;
                      }
                    }
// Calculate diffuse shading
//         float4 N = get_normal(&primitives[prim_index], point_intersect);
// TODO: review mod: previous line to next 2 lines:
                    Primitive pp = primitives[prim_index];
                    float4 N = get_normal(&pp, point_intersect);
                    if (primitives[prim_index].m_diff > 0) {
                      float dot_prod = DOT(N, L);
                      if (dot_prod > 0) {
                        float diff = dot_prod * primitives[prim_index].m_diff * shade;
                        ray_col += diff * primitives[prim_index].m_color * primitives[l].m_color;
                      }
                    }
// Calculate specular shading
                    if (primitives[prim_index].m_spec > 0) {
                      float4 V = cur_ray.direction;
                      float4 R = L - 1.5f * DOT(L, N) * N;
                      float dot_prod = DOT(V, R);
                      if (dot_prod > 0) {
// TODO: review, originally native_powr(dot_prod, 20)
                        float spec = cl::sycl::powr(dot_prod, 20.0f) * primitives[prim_index].m_spec * shade;
                        ray_col += spec * primitives[l].m_color;
                      }
                    }
                  }
                }
              }

            } // prim_index = -1
//return prim_index;
          }


// reflected/refracted rays have different modifiers on the color of the object
          switch (cur_ray.type) {
            case ORIGIN:acc += ray_col * cur_ray.weight;
              break;
            case REFLECTED:
              acc += ray_col * cur_ray.weight * primitives[cur_ray.origin_primitive].m_color *
                  cur_ray.transparency;
              break;
            case REFRACTED:acc += ray_col * cur_ray.weight * cur_ray.transparency;
              break;
          }
// handle reflection & refraction
          if (cur_ray.depth < TRACEDEPTH) {
// reflection
            float refl = primitives[prim_index].m_refl;
            if (refl > 0.0f) {
// float4 N = get_normal(&primitives[prim_index], point_intersect);
// TODO: review mod: previous line to next two lines:
              Primitive pp = primitives[prim_index];
              float4 N = get_normal(&pp, point_intersect);
              float4 R = cur_ray.direction - 2.0f * DOT(cur_ray.direction, N) * N;
              RayK new_ray;
              new_ray.origin = point_intersect + R * EPSILON;
              new_ray.direction = R;
              new_ray.depth = cur_ray.depth + 1;
              new_ray.weight = refl * cur_ray.weight;
              new_ray.type = REFLECTED;
              new_ray.origin_primitive = prim_index;
              new_ray.r_index = cur_ray.r_index;
              new_ray.transparency = cur_ray.transparency;
              PUSH_RAY(queue, new_ray, back_ray_ptr, rays_in_queue)
            }
// refraction
            float refr = primitives[prim_index].m_refr;
            if (refr > 0.0f) {
              float m_rindex = primitives[prim_index].m_refr_index;
              float n = cur_ray.r_index / m_rindex;
// float4 N = get_normal(&primitives[prim_index], point_intersect) * (float)result;
// TODO: review mod: previous line to next two lines:
              Primitive pp = primitives[prim_index];
              float4 N = get_normal(&pp, point_intersect) * (float) result;
              float cosI = -DOT(N, cur_ray.direction);
              float cosT2 = 1.0f - n * n * (1.0f - cosI * cosI);
              if (cosT2 > 0.0f) {
                float4 T = (n * cur_ray.direction) + (n * cosI - SQRT(cosT2)) * N;
                RayK new_ray;
                new_ray.origin = point_intersect + T * EPSILON;
                new_ray.direction = T;
                new_ray.depth = cur_ray.depth + 1;
                new_ray.weight = cur_ray.weight;
                new_ray.type = REFRACTED;
                new_ray.origin_primitive = prim_index;
                new_ray.r_index = m_rindex;
                new_ray.transparency =
                    cur_ray.transparency * (exp(primitives[prim_index].m_color * 0.15f * (-dist)));
                PUSH_RAY(queue, new_ray, back_ray_ptr, rays_in_queue)
              }
            }
          }
        }
      }

// Since we supersample 3x, we have to divide the total color by 9 to average it.
    uchar red = clamp(acc.x() * (256 / 9), 0.0f, 255.0f);
    uchar green = clamp(acc.y() * (256 / 9), 0.0f, 255.0f);
    uchar blue = clamp(acc.z() * (256 / 9), 0.0f, 255.0f);

// pixels[y * image_width + x] = (Pixel)(red, green, blue, 0);

// remove the offsets here
// int pos = y * image_width + x;
    const int x_w = (id) % image_width;
    const int y_w = (id) / image_width;
    int pos = y_w * image_width + x_w;
    pixels[pos] = Pixel{red, green, blue, 0};

// }); // end work-item

  });

});
}
